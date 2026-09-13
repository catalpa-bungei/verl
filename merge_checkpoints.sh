export PYTHONPATH=/mnt/shared-storage-gpfs2/evobox-share-gpfs2/yangxuqing/verl
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /mnt/shared-storage-gpfs2/evobox-share-gpfs2/yangxuqing/verl/checkpoints/verl_grpo_text-test0.01_qwen3-4b_promptv8_T5_temp0.7/text-qwen3-4b-RLCR_n5/global_step_103/actor \
    --target_dir /mnt/shared-storage-gpfs2/evobox-share-gpfs2/yangxuqing/verl/merged_model/qwen3-4b_promptv8_T5_temp0.7-text-RLCR_n5_20260715/global_step_103
