"""
Task 5's real held-out evaluation ("THE HARD FLOOR" per plan.md).

n=210 with a single 80/20 split would leave ~42 test rows spread across
three classes (benign/obvious/subtle) — too noisy at this sample size to
trust. Uses stratified 5-fold cross-validation instead (stratified on the
3-way benign/obvious/subtle label), aggregating out-of-fold predictions:
every one of the 210 rows is scored exactly once, by a model that never
saw it during training. That's a more robust "held-out, not training
data" result at this sample size than one lucky/unlucky split would give
— reported as such, not as a single number stripped of that context.

Only attack #1 (pre-signature Intent Artifact corruption) has generated
data to evaluate here — attack #2 (merchant-network laundering, Task 8's
GNN) has no data yet, so "per attack type" below means per subtlety
(obvious/subtle), the finest breakdown that's honestly available right
now, not a claim of covering both built attack types.
"""

import ast
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from defend.lightgbm_baseline import build_feature_matrix, train_baseline

_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "joined_sessions.csv")
_N_FOLDS = 5
_SEED = 42
_THRESHOLD = 0.5

_LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "num_leaves": 15,
    "min_data_in_leaf": 5,
}


def cross_validated_predictions(X: pd.DataFrame, y: pd.Series, strata: pd.Series, categorical_cols: list[str]) -> np.ndarray:
    """Out-of-fold predicted probabilities — every row scored by a model
    that never trained on it."""
    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=_N_FOLDS, shuffle=True, random_state=_SEED)
    for train_idx, test_idx in skf.split(X, strata):
        train_data = lgb.Dataset(
            X.iloc[train_idx], label=y.iloc[train_idx],
            categorical_feature=categorical_cols, free_raw_data=False,
        )
        model = lgb.train(_LGB_PARAMS, train_data, num_boost_round=100)
        oof[test_idx] = model.predict(X.iloc[test_idx])
    return oof


def _report(name: str, y_true: pd.Series, y_prob: np.ndarray) -> None:
    if y_true.nunique() < 2:
        print(f"{name}: n={len(y_true)} — only one class present, AUC undefined")
        return
    y_pred = (y_prob >= _THRESHOLD).astype(int)
    print(
        f"{name}: n={len(y_true)}  "
        f"precision={precision_score(y_true, y_pred, zero_division=0):.3f}  "
        f"recall={recall_score(y_true, y_pred, zero_division=0):.3f}  "
        f"F1={f1_score(y_true, y_pred, zero_division=0):.3f}  "
        f"AUC={roc_auc_score(y_true, y_prob):.3f}"
    )


def run() -> None:
    df = pd.read_csv(_DATA_PATH)
    subtlety = df["subtlety"]
    X, y, categorical_cols = build_feature_matrix(df)

    print("=" * 70)
    print(f"TASK 5 — LightGBM baseline, {X.shape[0]} rows x {X.shape[1]} features")
    print("CAVEAT: small-n-large-p (210 rows, 443 features after engineering).")
    print("Numbers below are a real first read of feasibility, not a")
    print("statistically bulletproof production evaluation — more generated")
    print("sessions (or eventually real telemetry) would tighten this.")
    print("=" * 70)

    oof_prob = cross_validated_predictions(X, y, subtlety, categorical_cols)

    print(f"\nEvaluation method: stratified {_N_FOLDS}-fold CV, out-of-fold predictions")
    print("(every row scored by a model that never trained on it)\n")

    _report("OVERALL", y, oof_prob)

    for label in ("obvious", "subtle"):
        mask = subtlety.isin([label, "benign"])
        _report(f"{label.upper()} vs benign", y[mask], oof_prob[mask])

    benign_mask = subtlety == "benign"
    benign_pred = (oof_prob[benign_mask] >= _THRESHOLD).astype(int)
    fp_rate = benign_pred.mean()
    print(f"\nFalse-positive rate on legitimate (benign) traffic: {fp_rate:.3f} "
          f"({benign_pred.sum()}/{benign_mask.sum()}) — TEAM_BRIEF.md Sec 4.8, non-negotiable")

    print("\n" + "-" * 70)
    print("Feature importance (gain, top 15) — from a single full-data fit,")
    print("NOT the cross-validated model above; for interpretation only:")
    model = train_baseline(X, y, categorical_cols)
    importance = pd.Series(model.feature_importance(importance_type="gain"), index=X.columns)
    print(importance.sort_values(ascending=False).head(15).to_string())
    print("-" * 70)


if __name__ == "__main__":
    run()
