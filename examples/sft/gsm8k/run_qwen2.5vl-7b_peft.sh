# Tested with 2 & 4 GPUs

set -x

# if [ "$#" -lt 2 ]; then
#     echo "Usage: run_qwen_05_peft.sh <nproc_per_node> <save_path> [other_configs...]"
#     exit 1
# fi

ENGINE=${1:-vllm}
export PYTHONPATH=/fs-computility/wangxuhong/yangxuqing/
export http_proxy=https://yangxuqing:Jf4r13R0xhV1QmLuDUoztEhzQS3fAAtkCB8Y97ypk5d0xTaO7H9hBiQFTCFL@volc-proxy.pjlab.org.cn:13128
export https_proxy=https://yangxuqing:Jf4r13R0xhV1QmLuDUoztEhzQS3fAAtkCB8Y97ypk5d0xTaO7H9hBiQFTCFL@volc-proxy.pjlab.org.cn:13128
# c2rm_train_path=/fs-computility/wangxuhong/yangxuqing/C2RM/data_C2RM/q/qwen7b/train_promptv7.parquet
# c2rm_test_path=/fs-computility/wangxuhong/yangxuqing/C2RM/data_C2RM/q/qwen7b/test_promptv7.parquet
c2rm_train_path=/fs-computility/wangxuhong/yangxuqing/C2RM/data_C2RM/q/qwen7b/train-mini.parquet
c2rm_test_path=/fs-computility/wangxuhong/yangxuqing/C2RM/data_C2RM/q/qwen7b/test-mini.parquet
train_files="$c2rm_train_path"
test_files="$c2rm_test_path"

nproc_per_node=1
save_path=/fs-computility/wangxuhong/yangxuqing/verl/output_sft

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=/fs-computility/ai-shen/shared/hf-hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/5b5eecc7efc2c3e86839993f2689bbbdf06bd8d4 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=c2rm-sft \
    trainer.experiment_name=c2rm-sft-qwen-2.5vl-7b-instruct \
    trainer.logger=['console'] \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null $@ \
    model.lora_rank=32\
    model.lora_alpha=16 \
    model.target_modules=all-linear

    # Or you can do this:
    # model.target_modules=[q_proj,v_proj] \
