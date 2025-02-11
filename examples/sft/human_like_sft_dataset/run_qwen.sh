set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer_hf \
    data.train_path=tuenguyen/human-like-sft-dataset-split \
    data.val_path=tuenguyen/human-like-sft-dataset-split \
    data.field_messages=messages \
    data.message_field_role=role \
    data.message_field_content=content \
    data.train_on_inputs=False \
    data.train_on_eos=all \
    data.sequence_len=1024 \
    data.chat_template=qwen_25 \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=human-like-sft-dataset-sft \
    trainer.experiment_name=human-like-sft-dataset-sft-qwen25 \
    trainer.total_epochs=4 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@