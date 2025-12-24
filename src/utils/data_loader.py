"""
Dataset Loading Utilities

Author: Rudra Sarker
Email: rudrasarker130@gmail.com
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_frozen_expert_dataset(data_path="data/raw/dataset_rl_distill.csv"):
    """
    Load frozen expert dataset
    
    Args:
        data_path: Path to dataset CSV file
        
    Returns:
        df: Pandas DataFrame with dataset
    """
    df = pd.read_csv(data_path)
    return df


def load_processed_tables(processed_dir="data/processed/"):
    """
    Load all processed CSV tables
    
    Args:
        processed_dir: Directory containing processed tables
        
    Returns:
        tables: Dictionary of DataFrames
    """
    processed_path = Path(processed_dir)
    
    tables = {
        'statistics': pd.read_csv(processed_path / 'table1_statistics.csv'),
        'energy_modes': pd.read_csv(processed_path / 'table2_energy_modes.csv'),
        'oracle_perf': pd.read_csv(processed_path / 'table3_oracle_perf.csv'),
        'robustness': pd.read_csv(processed_path / 'table4_robustness.csv')
    }
    
    return tables


def extract_features_targets(df):
    """
    Extract features and targets from dataset
    
    Args:
        df: Dataset DataFrame
        
    Returns:
        X: Features array
        y: Targets array (z1, z2)
    """
    # Extract observation features
    obs_cols = [col for col in df.columns if col.startswith('obs_')]
    X = df[obs_cols].values
    
    # Extract null-space coefficients
    y = df[['z1', 'z2']].values
    
    return X, y


def split_by_wind_speed(df, wind_threshold=10.0):
    """
    Split dataset by wind speed
    
    Args:
        df: Dataset DataFrame
        wind_threshold: Wind speed threshold (m/s)
        
    Returns:
        df_low_wind: Low wind subset
        df_high_wind: High wind subset
    """
    df_low_wind = df[df['wind_speed'] < wind_threshold]
    df_high_wind = df[df['wind_speed'] >= wind_threshold]
    
    return df_low_wind, df_high_wind