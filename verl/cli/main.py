import sys
import os
from omegaconf import OmegaConf
from typing import Optional


def train_ppo(config_path: str):
    """Run PPO/GRPO training with the given config file."""
    base_config_path = os.path.join(os.path.dirname(__file__), "base_ppo_trainer.yaml")

    # Load base config
    if not os.path.exists(base_config_path):
        print(f"Base config file not found: {base_config_path}")
        sys.exit(1)
    config = OmegaConf.load(base_config_path)


    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    user_config = OmegaConf.load(config_path)
    config = OmegaConf.merge(config, user_config)
    
    from verl.trainer.main_ppo import main as main_ppo
    main_ppo(config)

def main():
    """Main entry point for the CLI."""
    if len(sys.argv) < 2:
        print("Usage: verl <command> [args]")
        print("\nAvailable commands:")
        print("  train-ppo [config_path] - Train using PPO/GRPO algorithm")
        sys.exit(1)

    command = sys.argv[1]
    
    if command == "train-ppo":
        config_path = sys.argv[2] if len(sys.argv) > 2 else None
        train_ppo(config_path)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()