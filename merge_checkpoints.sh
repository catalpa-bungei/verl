export PYTHONPATH=/mnt/shared-storage-user/yangxuqing
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /mnt/shared-storage-user/yangxuqing/verl/checkpoints/verl_grpo_text-test0.01_qwen2.5vl-7b_promptv8_T5_temp0.7-group/text-beta0.5_20251226/global_step_103/actor \
    --target_dir /mnt/shared-storage-user/yangxuqing/verl/merged_model/qwen2.5vl-7b_promptv8_T5_temp0.7-text-beta0.5-group0.5_20251226/global_step_103 \
