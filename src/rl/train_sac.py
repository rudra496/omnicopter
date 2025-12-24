"""
SAC Training Script for OmniCopter Energy-Aware Null-Space Control

Author: Rudra Sarker
Email: rudrasarker130@gmail.com
"""

import yaml
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from src.utils.allocation import AllocationModel
from src.utils.energy_proxy import power_proxy


def load_config(config_path="configs/sac_params.yaml"):
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_omav_environment(config):
    """Create OMAV simulation environment"""
    # Import here to avoid circular imports
    from omav_env import OMAVEnv
    
    env_kwargs = {
        'allocation_model': AllocationModel(),
        'wind_params': config.get('wind_params', {}),
        'domain_randomization': config.get('domain_randomization', True),
        'reward_weights': config.get('reward_weights', {})
    }
    
    return OMAVEnv(**env_kwargs)


def main():
    """Main training function"""
    print("=" * 60)
    print("OmniCopter - SAC Expert Training")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    print(f"Configuration loaded from: {config.get('config_path', 'configs/sac_params.yaml')}")
    
    # Create environment
    print("Creating OMAV environment...")
    env = create_omav_environment(config)
    
    # Initialize SAC agent
    print("Initializing SAC agent...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=config.get('learning_rate', 3e-4),
        buffer_size=config.get('buffer_size', 1000000),
        batch_size=config.get('batch_size', 256),
        tau=config.get('tau', 0.005),
        gamma=config.get('gamma', 0.99),
        verbose=1,
        tensorboard_log="results/logs/"
    )
    
    # Train the agent
    total_timesteps = config.get('total_timesteps', 1000000)
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, log_interval=10)
    
    # Save the trained model
    model.save("results/models/sac_expert")
    print("Training complete. Model saved to 'results/models/sac_expert'")
    
    return model


if __name__ == "__main__":
    main()