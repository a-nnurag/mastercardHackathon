"""
LightGBM baseline — Defend layer 2, per plan.md Task 5 ("THE HARD FLOOR").

Feature matrix built from data/joined_sessions.csv (Task 4's fabricated
join of generated AgentSessions onto real IEEE-CIS transaction rows).
Held-out evaluation (cross-validated, not scored on training data) lives
in defend/evaluation.py — this module only builds features and fits one
model on all data, for the feature-importance report.
"""

import ast
import os
from typing import Any, Optional

import lightgbm as lgb
import pandas as pd

from defend.constraint_drift import extract_domain

_DROP_COLUMNS = [
    "agent_id",         # unique identifier, no signal
    "TransactionID",    # unique identifier, no signal
    "subtlety",         # leaks the label: subtlety=="benign" => injection_present==False
    "injection_present",  # the target itself, becomes y
    "ieee_cis_isFraud",   # constant (0) by construction
    # Defense signals excluded from tabular baseline to prevent circular label leakage:
    "constraint_drift",
    "ingestion_source_trust_score",
    "utterance_artifact_divergence",
    "lexical_divergence",
    "semantic_divergence",
    "content_injection",
    "content_score",
    "injection_payload_text",
    "raw_utterance",
    "signed_artifact_text",
    "intent_artifact_hash",
]

_CATEGORICAL_COLUMNS = [
    "agent_registry_status", "mandate_category", "task_origin_domain",
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_12", "id_15", "id_16", "id_28", "id_29", "id_30", "id_31",
    "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo",
]


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Returns (X, y, categorical_cols). See module + plan for the
    rationale behind each drop/engineered column."""
    df = df.copy()
    y = df["injection_present"].astype(int)

    # mandate_categories / mandate_merchant_allowlist / content_sources_ingested
    # may be Python lists or CSV string reprs — parse safely for both.
    def _parse_list(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                return [val]
        return val if isinstance(val, list) else []

    categories = df["mandate_categories"].apply(_parse_list)
    allowlist = df["mandate_merchant_allowlist"].apply(_parse_list)
    sources = df["content_sources_ingested"].apply(_parse_list)

    df["mandate_category"] = categories.apply(lambda c: c[0] if c else None)
    df["mandate_allowlist_size"] = allowlist.apply(len)
    df["task_origin_domain"] = df["task_origin_url"].apply(extract_domain)
    df["num_content_sources"] = sources.apply(len)

    drop_cols = [
        c for c in _DROP_COLUMNS + [
            "mandate_categories", "mandate_merchant_allowlist",
            "content_sources_ingested", "task_origin_url",
        ] if c in df.columns
    ]
    df = df.drop(columns=drop_cols)

    categorical_cols = [c for c in _CATEGORICAL_COLUMNS if c in df.columns]
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    # Safety net, not just the hardcoded list above: IEEE-CIS's identity
    # columns (id_12..id_38) are so sparse in the original 210-row sample
    # that several of them (id_23, id_27, at least) were all-NaN there and
    # never showed up as a non-numeric dtype -- so they weren't in
    # _CATEGORICAL_COLUMNS. Task 10's Mutator sampled fresh IEEE-CIS rows
    # and immediately hit real string values in those same columns,
    # crashing LightGBM ("pandas dtypes must be int, float or bool"). A
    # hardcoded list can always miss a column a new sample happens to
    # populate; casting every remaining non-numeric column closes that
    # gap generally. Checked against pandas.api.types rather than
    # `dtype == object`: this pandas version (3.0.5) defaults string
    # columns to its own StringDtype, not legacy `object` — a mixed
    # float64-NaN/string column from a concat DOES still come out as
    # `object` (that's the exact shape the real bug took), but a
    # same-dtype-throughout string column would not, and `== object`
    # alone would silently miss it.
    remaining_cols = [
        c for c in df.columns
        if c not in categorical_cols and not pd.api.types.is_numeric_dtype(df[c])
    ]
    for col in remaining_cols:
        df[col] = df[col].astype("category")
    categorical_cols = categorical_cols + remaining_cols

    return df, y, categorical_cols


def _encode_categoricals(X: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    import numpy as np
    X_num = X.copy()
    for col in categorical_cols:
        if col in X_num.columns:
            if hasattr(X_num[col], "cat"):
                X_num[col] = X_num[col].cat.codes.astype(np.float64)
            else:
                X_num[col] = pd.Categorical(X_num[col]).codes.astype(np.float64)
    # Ensure remaining columns are also numeric float64
    for col in X_num.columns:
        if not pd.api.types.is_numeric_dtype(X_num[col]):
            X_num[col] = pd.Categorical(X_num[col]).codes.astype(np.float64)
        else:
            X_num[col] = X_num[col].astype(np.float64)
    return X_num


def train_baseline(X: pd.DataFrame, y: pd.Series, categorical_cols: list[str]) -> Any:
    """Trains a gradient boosting classifier for tabular session fraud detection."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X_encoded = _encode_categoricals(X, categorical_cols)
    clf = HistGradientBoostingClassifier(
        max_iter=50,
        min_samples_leaf=1,
        learning_rate=0.1,
        random_state=42,
    )
    clf.fit(X_encoded.values, y.values)
    return clf


def save_model(model: Any, path: str) -> None:
    """Persists a trained model to disk."""
    import os, joblib
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path)


def _create_synthetic_joined_dataset() -> pd.DataFrame:
    """Creates a synthetic joined dataset from synthetic_sessions if IEEE-CIS join is not present."""
    from defend.constraint_drift import compute_constraint_drift, compute_ingestion_source_trust_score
    from generate.synthetic_sessions import all_sessions

    sessions = all_sessions()
    rows = []
    for i, s in enumerate(sessions):
        row = s.to_row()
        row["subtlety"] = "obvious" if s.injection_present else "benign"
        row["constraint_drift"] = compute_constraint_drift(s)
        row["ingestion_source_trust_score"] = compute_ingestion_source_trust_score(s)
        row["utterance_artifact_divergence"] = 0.8 if s.injection_present else 0.1
        row["TransactionID"] = 1000 + i
        row["ieee_cis_isFraud"] = 0
        row["ProductCD"] = "W"
        row["card4"] = "mastercard"
        row["card6"] = "credit"
        rows.append(row)
    return pd.DataFrame(rows)


def load_model(path: str) -> Any:
    """Loads a serialized model artifact from disk.
    
    Invariants:
    - Pure loader: never trains or creates models at runtime.
    - If the file is missing or unreadable, raises FileNotFoundError.
    """
    import os, joblib
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found at '{path}'. Train offline via defend.training.pipeline.")
    return joblib.load(path)


def predict_probability(model: Any, feature_row: pd.DataFrame) -> float:
    """Pure inference helper — predicts probability on a feature row."""
    cat_cols = [c for c in feature_row.columns if hasattr(feature_row[c], "cat") or not pd.api.types.is_numeric_dtype(feature_row[c])]
    num_row = _encode_categoricals(feature_row, cat_cols)
    if hasattr(model, "predict_proba"):
        preds = model.predict_proba(num_row.values)[:, 1]
    else:
        preds = model.predict(num_row.values)
    return float(preds[0])


from typing import Optional
from defend.contracts import SignalResult
from defend.detection_context import DetectionContext
from defend.signals.base import SignalDetector
from generate.session_schema import AgentSession


class LightGBMDetector(SignalDetector):
    name: str = "lightgbm_prob"
    supported_attack_families: tuple[str, ...] = ("prompt_injection",)

    def __init__(self, model: Optional[lgb.Booster] = None):
        self.model = model

    def detect(
        self,
        session: AgentSession,
        context: Optional[DetectionContext] = None,
    ) -> SignalResult:
        model = self.model
        if model is None and context is not None:
            model = context.get_model("lightgbm")

        if model is None:
            return SignalResult(
                name=self.name,
                value=None,
                available=False,
                evidence=[],
                hard_violation=False,
                metadata={"reason": "lightgbm_model_unavailable"},
            )

        feature_row = None
        if context is not None and "feature_matrix" in context.metadata:
            agent_id_to_row = context.metadata.get("agent_id_to_row", {})
            X = context.metadata["feature_matrix"]
            if session.agent_id in agent_id_to_row:
                feature_row = X.iloc[[agent_id_to_row[session.agent_id]]]

        if feature_row is None:
            # If no precomputed feature row available, return unavailable gracefully
            return SignalResult(
                name=self.name,
                value=None,
                available=False,
                evidence=[],
                hard_violation=False,
                metadata={"reason": "feature_row_not_in_context"},
            )

        prob = predict_probability(model, feature_row)
        evidence = []
        if prob >= 0.5:
            evidence.append(f"lightgbm_high_risk_probability: score {prob:.3f} >= 0.50")
        elif prob >= 0.15:
            evidence.append(f"lightgbm_medium_risk_probability: score {prob:.3f} >= 0.15")

        return SignalResult(
            name=self.name,
            value=prob,
            available=True,
            evidence=evidence,
            hard_violation=False,
            metadata={
                "probability": prob,
                "model_role": "behavioral_transaction_baseline",
            },
        )


if __name__ == "__main__":
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "joined_sessions.csv")
    df = pd.read_csv(path)
    X, y, categorical_cols = build_feature_matrix(df)
    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} columns ({len(categorical_cols)} categorical)")
    model = train_baseline(X, y, categorical_cols)
    print("Trained one full-data model (for feature importance — see defend/evaluation.py for real held-out metrics)")
