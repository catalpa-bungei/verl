python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /fs-computility/wangxuhong/yangxuqing/verl/checkpoints/verl_grpo_text-test0.01_llama-3.2-3b-Instruct_promptv8_T5_temp0.7/text-beta0.1_alpha0.5/global_step_103/actor \
    --target_dir /fs-computility/wangxuhong/yangxuqing/verl/merged_model/llama-3.2-3b_promptv8_T5_temp0.7-text-beta0.1_alpha0.5/global_step_103 \