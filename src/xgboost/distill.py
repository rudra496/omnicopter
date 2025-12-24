"""
XGBoost Distillation Script

Author: Rudra Sarker
Email: rudrasarker130@gmail.com
"""

import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time


def load_config(config_path="configs/xgboost_params.yaml"):
    """Load XGBoost configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(data_path):
    """Load frozen expert dataset"""
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    return df


def prepare_data(df, config):
    """Prepare features and targets"""
    # Extract observation features (21 dimensions)
    obs_cols = [col for col in df.columns if col.startswith('obs_')]
    X = df[obs_cols].values
    
    # Extract null-space coefficients
    y1 = df['z1'].values
    y2 = df['z2'].values
    
    # Train-test split
    test_size = config.get('test_size', 0.2)
    random_state = config.get('random_state', 42)
    
    X_train, X_test, y1_train, y1_test = train_test_split(
        X, y1, test_size=test_size, random_state=random_state
    )
    
    _, _, y2_train, y2_test = train_test_split(
        X, y2, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y1_train, y1_test, y2_train, y2_test


def train_oracle(X_train, y_train, config):
    """Train XGBoost oracle"""
    params = {
        'objective': 'reg:squarederror',
        'max_depth': config.get('max_depth', 6),
        'learning_rate': config.get('learning_rate', 0.1),
        'n_estimators': config.get('n_estimators', 100),
        'subsample': config.get('subsample', 0.8),
        'colsample_bytree': config.get('colsample_bytree', 0.8),
        'random_state': config.get('random_state', 42)
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    return model


def evaluate_oracle(model, X_test, y_test):
    """Evaluate oracle performance"""
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    return r2, mse, rmse, y_pred


def benchmark_inference(model, X_test, n_samples=1000):
    """Benchmark inference latency"""
    X_sample = X_test[:n_samples]
    
    start_time = time.time()
    _ = model.predict(X_sample)
    end_time = time.time()
    
    latency_ms = (end_time - start_time) / n_samples * 1000
    
    return latency_ms


def main():
    """Main distillation function"""
    print("=" * 60)
    print("OmniCopter - XGBoost Oracle Distillation")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Load dataset
    data_path = config.get('data_path', 'data/raw/dataset_rl_distill.csv')
    df = load_dataset(data_path)
    
    # Prepare data
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = prepare_data(df, config)
    
    # Train oracle for z1
    print("\nTraining oracle for z1...")
    oracle_z1 = train_oracle(X_train, y1_train, config)
    r2_z1, mse_z1, rmse_z1, _ = evaluate_oracle(oracle_z1, X_test, y1_test)
    print(f"z1 Oracle - R²: {r2_z1:.4f}, RMSE: {rmse_z1:.4f}")
    
    # Train oracle for z2
    print("\nTraining oracle for z2...")
    oracle_z2 = train_oracle(X_train, y2_train, config)
    r2_z2, mse_z2, rmse_z2, _ = evaluate_oracle(oracle_z2, X_test, y2_test)
    print(f"z2 Oracle - R²: {r2_z2:.4f}, RMSE: {rmse_z2:.4f}")
    
    # Benchmark inference
    print("\nBenchmarking inference latency...")
    latency_z1 = benchmark_inference(oracle_z1, X_test)
    latency_z2 = benchmark_inference(oracle_z2, X_test)
    avg_latency = (latency_z1 + latency_z2) / 2
    print(f"Average inference latency: {avg_latency:.4f} ms/sample")
    
    # Save models
    oracle_z1.save_model("results/models/oracle_z1.json")
    oracle_z2.save_model("results/models/oracle_z2.json")
    print("\nModels saved to results/models/")
    
    return oracle_z1, oracle_z2


if __name__ == "__main__":
    main()