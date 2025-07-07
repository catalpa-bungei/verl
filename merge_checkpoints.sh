python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /fs-computility/wangxuhong/yangxuqing/verl/checkpoints/verl_grpo_c2rm-test0.01_qwen2.5vl-7b_promptv7_T5_temp0.7/c2rm-beta0.1_ece_we0.1_alpha0.5/global_step_162/actor \
    --target_dir /fs-computility/wangxuhong/yangxuqing/verl/merged_model/qwen2_5_vl_7b_promptv7_T5_temp0.7-beta0.1_ece_we0.1_alpha0.5/global_step_162 \