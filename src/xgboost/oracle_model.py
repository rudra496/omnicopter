"""
XGBoost Oracle Model Wrapper

Author: Rudra Sarker
Email: rudrasarker130@gmail.com
"""

import xgboost as xgb
import numpy as np


class OracleModel:
    """Wrapper for XGBoost oracle models"""
    
    def __init__(self, model_z1_path="results/models/oracle_z1.json", 
                 model_z2_path="results/models/oracle_z2.json"):
        """
        Initialize oracle models
        
        Args:
            model_z1_path: Path to z1 oracle model
            model_z2_path: Path to z2 oracle model
        """
        self.oracle_z1 = xgb.XGBRegressor()
        self.oracle_z1.load_model(model_z1_path)
        
        self.oracle_z2 = xgb.XGBRegressor()
        self.oracle_z2.load_model(model_z2_path)
        
    def predict(self, observation):
        """
        Predict null-space coefficients from observation
        
        Args:
            observation: Environment observation (21 dimensions)
            
        Returns:
            coefficients: [z1, z2] null-space coefficients
        """
        obs = np.array(observation).reshape(1, -1)
        
        z1 = self.oracle_z1.predict(obs)[0]
        z2 = self.oracle_z2.predict(obs)[0]
        
        return np.array([z1, z2])
    
    def predict_batch(self, observations):
        """
        Predict null-space coefficients for batch of observations
        
        Args:
            observations: Batch of environment observations
            
        Returns:
            coefficients: Array of [z1, z2] pairs
        """
        obs = np.array(observations)
        
        z1_pred = self.oracle_z1.predict(obs)
        z2_pred = self.oracle_z2.predict(obs)
        
        return np.column_stack([z1_pred, z2_pred])