set -x

# gsm8k_train_path=$HOME/data/gsm8k/train.parquet
# gsm8k_test_path=$HOME/data/gsm8k/test.parquet
# math_train_path=$HOME/data/math/train.parquet
# math_test_path=$HOME/data/math/test.parquet

gsm8k_train_path=/fs-computility/ai-shen/yangxuqing/verl/data/gsm8k/train.parquet
gsm8k_test_path=/fs-computility/ai-shen/yangxuqing/verl/data/gsm8k/test.parquet
math_train_path=/fs-computility/ai-shen/yangxuqing/verl/data/math/train.parquet
math_test_path=/fs-computility/ai-shen/yangxuqing/verl/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

export WANDB_API_KEY=f49497a793fd30f43cd1d8279cde35b43c3dd7c8

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/fs-computility/ai-shen/shared/hf-hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.96 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=/fs-computility/ai-shen/shared/hf-hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_example_qwen2.5-32b-batch128' \
    trainer.experiment_name='Qwen2.5-32B-Instruct_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 $@
