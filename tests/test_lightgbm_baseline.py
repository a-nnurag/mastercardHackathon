import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from defend.lightgbm_baseline import build_feature_matrix

_REQUIRED_COLS = {
    "agent_id": "agent-1", "TransactionID": 1, "subtlety": "benign", "injection_present": False,
    "ieee_cis_isFraud": 0, "mandate_categories": "['travel']", "mandate_merchant_allowlist": "['cleartrip.com']",
    "content_sources_ingested": "['cleartrip.com']", "task_origin_url": "https://cleartrip.com",
    "agent_registry_status": "valid", "utterance_artifact_divergence": 0.1, "constraint_drift": 0.0,
    "ingestion_source_trust_score": 0.0, "hops_since_intent": 0, "tool_calls_made": 3,
}


def test_object_dtype_column_not_in_hardcoded_list_still_gets_cast():
    # Regression test for a real bug found via Task 10's Mutator: a fresh
    # IEEE-CIS sample populated id_23/id_27 with real strings, columns
    # that weren't in _CATEGORICAL_COLUMNS because the original 210-row
    # sample happened to have them all-NaN. LightGBM crashes on raw
    # object-dtype columns -- build_feature_matrix must catch any column
    # like this, not just the ones on the hardcoded list.
    row = dict(_REQUIRED_COLS)
    row["id_23"] = "IP_PROXY:TRANSPARENT"  # not in _CATEGORICAL_COLUMNS
    df = pd.DataFrame([row, {**row, "agent_id": "agent-2", "id_23": "IP_PROXY:ANONYMOUS"}])

    X, y, categorical_cols = build_feature_matrix(df)

    assert "id_23" in categorical_cols
    assert str(X["id_23"].dtype) == "category"


def test_known_categorical_columns_still_handled():
    row = dict(_REQUIRED_COLS)
    row["ProductCD"] = "W"
    df = pd.DataFrame([row, {**row, "agent_id": "agent-2", "ProductCD": "C"}])

    X, y, categorical_cols = build_feature_matrix(df)

    assert "ProductCD" in categorical_cols
    assert str(X["ProductCD"].dtype) == "category"
