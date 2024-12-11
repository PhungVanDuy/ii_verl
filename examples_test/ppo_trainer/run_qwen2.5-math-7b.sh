set -x

math_train_path=/workspace/ii_verl/examples_test/ppo_trainer/master_train_v2.parquet
math_test_path=/workspace/ii_verl/examples_test/ppo_trainer/master_test_v2.parquet

train_files="['$math_train_path']"
test_files="['$math_test_path']"

python3 -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512\
    data.val_batch_size=256\
    data.max_prompt_length=512 \
    data.max_response_length=1280\
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-7B-Instruct\
    actor_rollout_ref.model.enable_gradient_checkpointing=True\
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128\
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16\
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    critic.optim.lr=1e-5 \
    critic.model.path=Qwen/Qwen2.5-Math-7B-Instruct\
    critic.model.enable_gradient_checkpointing=True\
    critic.ppo_micro_batch_size=16 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tracking'] \
    trainer.project_name='verl_example' \
    trainer.experiment_name='math_qwen2.5_math_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 $@
