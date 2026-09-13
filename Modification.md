# Modification

## Varying Trajectory Number Per Sample

The customized GRPO filtering in `verl/trainer/ppo/ray_trainer.py` now supports keeping a variable number of trajectories for each prompt/sample.

- Adaptive filtering keeps trajectories through the first failed confidence check:
  `indices[:cutoff_idx + 1]`.
- This means one `uid` group may keep fewer than `rollout.n` trajectories.
- GRPO remains compatible because advantage computation groups samples by `uid`, not by fixed-size chunks of `rollout.n`.
- Group statistics such as `group_avg_acc` and `group_correlation` are recomputed after filtering, so reward computation sees the truncated group actually used for training.
- Once the accumulated filtered batch reaches the target size, the trainer slices by raw sample count.
- This raw slice may cut through the middle of a `uid` group. This is intentional in the current implementation.
- No fake/default trajectories are added. If there are more real kept trajectories than needed, the remainder is carried over to the next step.

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

### Example: Raw Slicing May Split a GRPO Group

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

This intentionally splits `uid C`, leaving `C1 C2 C3` outside this PPO update:

```text
training batch: [A0, A1, A2, B0, B1, C0]
carry over:     [C1, C2, C3, D0]
```

After slicing, `group_avg_acc` and `group_correlation` are recomputed on the sliced training batch. Therefore `C0` is treated according to the samples present in the current PPO update, and the carried-over `C1 C2 C3` will be handled in a later update.

### Example: Batch Size Reached

The trainer does not add default or fake trajectories. It slices as soon as the accumulated filtered batch reaches the target size.

Example:

```text
target_size = 64
current_size = 70
```

The trainer takes:

```python
actual_batch_size = min(current_size, target_size)
batch = accumulated_batch[:actual_batch_size]
```

So the current PPO update receives the first `64` real trajectories, and the remaining `6` real trajectories are carried over:

```text
training batch size: 64
carried over:        6
```
