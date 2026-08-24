"""
Offline Training Pipeline Entrypoint.

Trains all models (LightGBM baseline, Merchant GNN) and persists artifacts
to artifacts/models/ for runtime consumption.
"""

import json
import os
from defend.training.lightgbm import train_and_save_lightgbm
from defend.training.gnn import train_and_save_gnn
from defend.gnn import train_and_evaluate

_ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "artifacts")
METRICS_PATH = os.path.join(_ARTIFACTS_DIR, "metrics.json")


def run_training_pipeline():
    print("==================================================")
    print("Starting Offline Model Training Pipeline")
    print("==================================================")

    os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
    lgb_model = train_and_save_lightgbm()
    gnn_model = train_and_save_gnn()

    print("[Metrics] Computing offline evaluation metrics...")
    attack2_metrics = train_and_evaluate()

    # Tabular baseline held-out metrics
    try:
        from mutator.mutate import JOINED_PATH
        import pandas as pd
        from defend.lightgbm_baseline import _create_synthetic_joined_dataset, build_feature_matrix
        from defend.evaluation import cross_validated_predictions
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

        if os.path.exists(JOINED_PATH):
            df = pd.read_csv(JOINED_PATH)
        else:
            df = _create_synthetic_joined_dataset()

        X, y, cat_cols = build_feature_matrix(df)
        oof = cross_validated_predictions(X, y, df["subtlety"], cat_cols)
        yp = (oof >= 0.5).astype(int)
        attack1_metrics = {
            "overall": {
                "n": len(df),
                "precision": float(precision_score(y, yp, zero_division=0)),
                "recall": float(recall_score(y, yp, zero_division=0)),
                "f1": float(f1_score(y, yp, zero_division=0)),
                "auc": float(roc_auc_score(y, oof)),
            }
        }
    except Exception as e:
        attack1_metrics = {"error": str(e)}

    metrics_payload = {
        "attack1_prompt_injection": attack1_metrics,
        "attack2_merchant_laundering": attack2_metrics,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[Metrics] Saved offline metrics to {METRICS_PATH}")

    print("==================================================")
    print("Offline Model Training Pipeline Completed Successfully")
    print("Artifacts ready in artifacts/models/ and artifacts/metrics.json")
    print("==================================================")


if __name__ == "__main__":
    run_training_pipeline()
