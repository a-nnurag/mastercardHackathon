"""
LightGBM baseline — Defend layer 2, per plan.md Task 5 ("THE HARD FLOOR").

Feature matrix built from data/joined_sessions.csv (Task 4's fabricated
join of generated AgentSessions onto real IEEE-CIS transaction rows).
Held-out evaluation (cross-validated, not scored on training data) lives
in defend/evaluation.py — this module only builds features and fits one
model on all data, for the feature-importance report.
"""

import ast

import lightgbm as lgb
import pandas as pd

from defend.constraint_drift import extract_domain

_DROP_COLUMNS = [
    "agent_id",         # unique identifier, no signal
    "TransactionID",    # unique identifier, no signal
    "subtlety",         # leaks the label: subtlety=="benign" => injection_present==False
    "injection_present",  # the target itself, becomes y
    "ieee_cis_isFraud",   # constant (0) by construction — Task 4 only sampled non-fraud
    # rows — and not our target even if it weren't constant.
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
    # round-tripped through CSV as Python-list string reprs (e.g. "['travel']") —
    # parse before deriving anything from them.
    categories = df["mandate_categories"].apply(ast.literal_eval)
    allowlist = df["mandate_merchant_allowlist"].apply(ast.literal_eval)
    sources = df["content_sources_ingested"].apply(ast.literal_eval)

    df["mandate_category"] = categories.apply(lambda c: c[0] if c else None)
    df["mandate_allowlist_size"] = allowlist.apply(len)
    df["task_origin_domain"] = df["task_origin_url"].apply(extract_domain)
    df["num_content_sources"] = sources.apply(len)

    df = df.drop(columns=_DROP_COLUMNS + [
        "mandate_categories", "mandate_merchant_allowlist",
        "content_sources_ingested", "task_origin_url",
    ])

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


def train_baseline(X: pd.DataFrame, y: pd.Series, categorical_cols: list[str]) -> lgb.Booster:
    """One fit on all data — for feature-importance reporting only.
    Held-out precision/recall/F1/AUC come from evaluation.py's
    cross-validation, never from this model scored on its own training data."""
    train_data = lgb.Dataset(X, label=y, categorical_feature=categorical_cols, free_raw_data=False)
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "num_leaves": 15,   # small on purpose — 210 rows, 448 features, avoid overfitting a toy model further
        "min_data_in_leaf": 5,
    }
    return lgb.train(params, train_data, num_boost_round=100)


if __name__ == "__main__":
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "joined_sessions.csv")
    df = pd.read_csv(path)
    X, y, categorical_cols = build_feature_matrix(df)
    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} columns ({len(categorical_cols)} categorical)")
    model = train_baseline(X, y, categorical_cols)
    print("Trained one full-data model (for feature importance — see defend/evaluation.py for real held-out metrics)")
