# cd yangxuqing/verl

set -x
ENGINE=${1:-vllm}
export http_proxy=https://yangxuqing:Jf4r13R0xhV1QmLuDUoztEhzQS3fAAtkCB8Y97ypk5d0xTaO7H9hBiQFTCFL@volc-proxy.pjlab.org.cn:13128
export https_proxy=https://yangxuqing:Jf4r13R0xhV1QmLuDUoztEhzQS3fAAtkCB8Y97ypk5d0xTaO7H9hBiQFTCFL@volc-proxy.pjlab.org.cn:13128
c2rm_train_path=/fs-computility/ai-shen/yangxuqing/C2RM/data_C2RM/q/qwen7b/train.parquet
c2rm_test_path=/fs-computility/ai-shen/yangxuqing/C2RM/data_C2RM/q/qwen7b/test.parquet
train_files="['$c2rm_train_path']"
test_files="['$c2rm_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=/fs-computility/ai-shen/shared/hf-hub/models--Qwen--Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=400 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    custom_reward_function.path=/fs-computility/ai-shen/yangxuqing/verl/verl/utils/reward_score/c2rm_reward.py \
    custom_reward_function.name=compute_score_reference_data \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_c2rm-display' \
    trainer.experiment_name='qwen2_5_vl_7b_c2rm-beta0.1-alpha0-eta0' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.total_epochs=1 $@