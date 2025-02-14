

data="tuenguyen/ultrachat_200k_subset"
train_column="train"
val_column="test"
model="Qwen/Qwen2.5-1.5B"
data="tuenguyen/open-r1-math-220k-chatml"

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
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$data \
    data.val_files=$data \
    data.truncation="right" \
    data.messages_key=messages \
    data.is_hf_dataset=True \
    data.use_multiturn=True \
    +data.key_train=$train_column\
    +data.key_val=$val_column\
    data.max_length=8192 \
    data.chat_template=tokenizer_default \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=$model \
    model.enable_gradient_checkpointing=False \
    model.use_liger=True \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=True \
    optim.lr=5e-6 \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.1 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=ultra_chat_sft_verl \
    trainer.experiment_name=math_sft_7b_ulysses_sequence_parallel_size_2 \
    trainer.total_epochs=4 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@


    # ulysses_sequence_parallel_size=2 \
    # use_remove_padding=True \