# Modification

## Varying Trajectory Number Per Sample

The customized GRPO filtering in `verl/trainer/ppo/ray_trainer.py` now supports keeping a variable number of trajectories for each prompt/sample.

- Adaptive filtering keeps trajectories through the first failed confidence check:
  `indices[:cutoff_idx + 1]`.
- This means one `uid` group may keep fewer than `rollout.n` trajectories.
- GRPO remains compatible because advantage computation groups samples by `uid`, not by fixed-size chunks of `rollout.n`.
- Group statistics such as `group_avg_acc` and `group_correlation` are recomputed after filtering, so reward computation sees the truncated group actually used for training.
- Accumulated batches are popped only at whole-`uid` group boundaries. This avoids splitting one prompt group across two PPO updates.
- The final training batch must also satisfy framework batch-size constraints, including DP partitioning, actor PPO mini-batch size, and log-prob micro-batch sizes.
- No fake/default trajectories are added. If there are not enough real kept trajectories to form a compatible batch, the trainer keeps accumulating until it can form one, or skips the update after the generation limit.

### Strict DAPO Filter

The correctness filter is strict:

```python
keep_indices_acc = np.where((group_avg_acc > 0.0) & (group_avg_acc < 1.0))[0]
```

This intentionally removes all-correct groups (`group_avg_acc == 1.0`) and all-wrong groups (`group_avg_acc == 0.0`), matching the DAPO-style idea of keeping groups with non-zero outcome variance.

### Example: Keeping Trajectories Until First False

If `rollout.n = 4` and adaptive sampling returns `cutoff_idx = 2`, the trainer keeps:

```text
uid A: A0 A1 A2
```

It drops only the later trajectories:

```text
dropped: A3
```

The first failed trajectory is included because the cutoff happens after observing it:

```python
keep_indices_adaptive.extend(indices[: cutoff_idx + 1])
```

### Example: Avoid Splitting a GRPO Group

After adaptive filtering, each prompt may keep a different number of trajectories:

```text
uid A: A0 A1 A2        # kept 3
uid B: B0 B1           # kept 2
uid C: C0 C1 C2 C3     # kept 4
uid D: D0              # kept 1
```

The accumulated batch is ordered as:

```text
[A0, A1, A2, B0, B1, C0, C1, C2, C3, D0]
```

If `target_size = 6`, a raw slice would produce:

```text
[A0, A1, A2, B0, B1, C0]
```

That would split `uid C`, leaving `C1 C2 C3` outside this PPO update. Since GRPO advantage is grouped by `uid`, this would make `C0` behave like a singleton group in the current update.

Instead, the trainer pops only at whole-`uid` boundaries. Valid boundaries in this example are:

```text
end = 3  -> [A0, A1, A2]
end = 5  -> [A0, A1, A2, B0, B1]
end = 9  -> [A0, A1, A2, B0, B1, C0, C1, C2, C3]
end = 10 -> all groups
```

For `target_size = 6`, it may choose `end = 5` instead of cutting at `6`:

```text
training batch: [A0, A1, A2, B0, B1]
carry over:     [C0, C1, C2, C3, D0]
```

### Example: Batch Divisor Compatibility

The trainer does not add default or fake trajectories. It waits until it can form a real batch whose size is compatible with downstream splitting.

Example:

```text
world_size = 8
rollout.n = 4
actor.ppo_mini_batch_size = 16 prompts
actor update mini-batch after rollout = 16 * 4 = 64 trajectories
rollout log_prob_micro_batch_size = 32
ref log_prob_micro_batch_size = 32
```

The required divisor is:

```text
lcm(8, 64, 32, 32) = 64
```

So the final filtered training batch must contain `64`, `128`, `192`, ... real trajectories. If the accumulated filtered data has `70` trajectories, the trainer looks for the largest prefix that:

- ends at a whole-`uid` boundary,
- is not larger than the target size,
- has length divisible by `64`.

If no such prefix exists yet, the trainer continues accumulating more real filtered trajectories. If the generation limit is reached and no compatible prefix can be formed, the update is skipped rather than padded with fake data.
