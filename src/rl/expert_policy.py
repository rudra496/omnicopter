"""
Frozen Expert Policy Module

Author: Rudra Sarker
Email: rudrasarker130@gmail.com
"""

import torch
import numpy as np
from stable_baselines3 import SAC


class ExpertPolicy:
    """Wrapper for frozen SAC expert policy"""
    
    def __init__(self, model_path="results/models/sac_expert"):
        """
        Initialize expert policy
        
        Args:
            model_path: Path to saved SAC model
        """
        self.model = SAC.load(model_path)
        self.model.policy.eval()
        
    def predict(self, observation, deterministic=True):
        """
        Predict action from observation
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Predicted null-space coefficients
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def predict_batch(self, observations, deterministic=True):
        """
        Predict actions for batch of observations
        
        Args:
            observations: Batch of environment observations
            deterministic: Whether to use deterministic policy
            
        Returns:
            actions: Predicted null-space coefficients
        """
        actions = []
        for obs in observations:
            action = self.predict(obs, deterministic=deterministic)
            actions.append(action)
        return np.array(actions)