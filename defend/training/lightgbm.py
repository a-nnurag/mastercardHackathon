"""
Offline training script for LightGBM baseline model.
"""

import os
import pandas as pd
from defend.lightgbm_baseline import (
    _create_synthetic_joined_dataset,
    build_feature_matrix,
    save_model,
    train_baseline,
)

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "joined_sessions.csv")
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "artifacts", "models", "lightgbm.txt")


def train_and_save_lightgbm(data_path: str = DEFAULT_DATA_PATH, model_save_path: str = DEFAULT_MODEL_PATH):
    if os.path.exists(data_path):
        print(f"[LightGBM] Loading training dataset from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print(f"[LightGBM] Training dataset not found at {data_path}, building synthetic baseline dataset...")
        df = _create_synthetic_joined_dataset()

    X, y, categorical_cols = build_feature_matrix(df)
    print(f"[LightGBM] Training feature matrix shape: {X.shape} ({len(categorical_cols)} categoricals)")

    model = train_baseline(X, y, categorical_cols)
    save_model(model, model_save_path)
    print(f"[LightGBM] Model artifact saved to {model_save_path}")
    return model


if __name__ == "__main__":
    train_and_save_lightgbm()
