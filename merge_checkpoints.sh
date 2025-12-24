export PYTHONPATH=/mnt/shared-storage-user/yangxuqing
python scripts/model_merger.py merge \
    --backend fsdp \
<<<<<<< HEAD
    --local_dir /fs-computility/wangxuhong/yangxuqing/verl/checkpoints/verl_grpo_text-test0.01_llama-3.2-3b-Instruct_promptv8_T5_temp0.7/text-beta0.1_alpha0.5/global_step_103/actor \
    --target_dir /fs-computility/wangxuhong/yangxuqing/verl/merged_model/llama-3.2-3b_promptv8_T5_temp0.7-text-beta0.1_alpha0.5/global_step_103 \
=======
    --local_dir /mnt/shared-storage-user/yangxuqing/verl/checkpoints/verl_grpo_text-test0.01_qwen2.5vl-7b_promptv8_T5_temp0.7/text-beta0.1_alpha0.5-20250920/global_step_103/actor \
    --target_dir /mnt/shared-storage-user/yangxuqing/verl/merged_model/qwen2.5vl-7b_promptv8_T5_temp0.7-text-beta0.1_alpha0.5-20250920/global_step_103 \
>>>>>>> a3ea79f409eddf1a8409819ec368de5c3de333ff
