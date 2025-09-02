# cd yangxuqing/verl

set -x
ENGINE=${1:-vllm}
export PYTHONPATH=/fs-computility/wangxuhong/yangxuqing/
export http_proxy=https://yangxuqing:Jf4r13R0xhV1QmLuDUoztEhzQS3fAAtkCB8Y97ypk5d0xTaO7H9hBiQFTCFL@volc-proxy.pjlab.org.cn:13128
export https_proxy=https://yangxuqing:Jf4r13R0xhV1QmLuDUoztEhzQS3fAAtkCB8Y97ypk5d0xTaO7H9hBiQFTCFL@volc-proxy.pjlab.org.cn:13128
# c2rm_train_path=/fs-computility/wangxuhong/yangxuqing/C2RM/data_C2RM/q/qwen7b/train_promptv7.parquet
# c2rm_test_path=/fs-computility/wangxuhong/yangxuqing/C2RM/data_C2RM/q/qwen7b/test_promptv7.parquet

train_path=/fs-computility/wangxuhong/yangxuqing/post_processing/create_training_data/data/text/text_train_promptv8.parquet
test_path=/fs-computility/wangxuhong/yangxuqing/post_processing/create_training_data/data/text/text_test_promptv8.parquet
balance_train_path=/fs-computility/wangxuhong/yangxuqing/post_processing/create_training_data/data/text/balance/text_balance_train_promptv8.parquet
balance_test_path=/fs-computility/wangxuhong/yangxuqing/post_processing/create_training_data/data/text/balance/text_balance_test_promptv8.parquet
proportion_train_path=/fs-computility/wangxuhong/yangxuqing/post_processing/create_training_data/data/text/proportion/text_proportion_train_promptv8.parquet
proportion_test_path=/fs-computility/wangxuhong/yangxuqing/post_processing/create_training_data/data/text/proportion/text_proportion_test_promptv8.parquet


train_files="['$train_path']"
test_files="['$test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.val_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=/fs-computility/ai-shen/shared/hf-hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/5b5eecc7efc2c3e86839993f2689bbbdf06bd8d4 \
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
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=500 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    custom_reward_function.path=/fs-computility/wangxuhong/yangxuqing/verl/verl/utils/reward_score/Brier_reward.py \
    custom_reward_function.name=compute_score_reference_data \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_text-test0.01_qwen2.5vl-7b_promptv8_T5_temp0.7' \
    trainer.experiment_name='text-brier1_alpha0.5' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.total_epochs=1 $@