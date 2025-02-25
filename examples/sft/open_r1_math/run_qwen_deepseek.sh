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
    data.train_path=tuenguyen/open-r1-math-220k-chatml-v2 \
    data.val_path=tuenguyen/open-r1-math-220k-chatml-v2 \
    data.field_messages=messages \
    data.message_field_role=role \
    data.message_field_content=content \
    data.train_on_inputs=True \
    data.train_on_eos=all \
    data.sequence_len=22400 \
    data.chat_template=tokenizer_default \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=Qwen/Qwen2.5-7B \
    model.use_liger=True \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=True \
    optim.lr=5e-6 \
    optim.weight_decay=0.0001 \
    optim.warmup_steps_ratio=0.0 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=open_r1_sft \
    trainer.experiment_name=open_r1_sft_ver8_base_model \
    trainer.total_epochs=4 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@


    # ulysses_sequence_parallel_size=2 \
    # use_remove_padding=True \