# VeRL CLI

## Setup Docker

```bash
bash docker/build.sh
bash docker/run.sh
```

## Train

Prepare config file as `examples_cli/qwen0.5b-grpo.yaml` or `examples_cli/qwen0.5b-ppo.yaml`

Serve a reward model server, follow example in `examples_reward_api` directory.

```bash
verl train-ppo <path_to_config_file>
```
