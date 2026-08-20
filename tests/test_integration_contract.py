import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jsonschema
import pytest

from defend.integration_contract import (
    SESSION_RISK_PAYLOAD_SCHEMA,
    _all_sessions_by_id,
    _classify_risk_level,
    _fit_production_model,
    build_session_risk_payload,
    measure_latency,
)
from defend.gnn import predict_all_merchants


def test_classify_risk_level_rules_flag_forces_high_regardless_of_score():
    assert _classify_risk_level(rules_flagged=True, score=0.01, content_score=0.0) == "HIGH"


def test_classify_risk_level_thresholds():
    assert _classify_risk_level(rules_flagged=False, score=0.6, content_score=0.0) == "HIGH"
    assert _classify_risk_level(rules_flagged=False, score=0.2, content_score=0.0) == "MEDIUM"
    assert _classify_risk_level(rules_flagged=False, score=0.0, content_score=0.2) == "MEDIUM"
    assert _classify_risk_level(rules_flagged=False, score=0.01, content_score=0.01) == "LOW"


@pytest.fixture(scope="module")
def production_model():
    # Real fit, real GNN training -- run once and shared, matching
    # tests/test_gnn.py's "exercise the real pipeline" style rather than
    # mocking a LightGBM model out of the equation.
    return _fit_production_model()


@pytest.fixture(scope="module")
def gnn_lookup():
    return predict_all_merchants()


def test_build_session_risk_payload_matches_json_schema(production_model, gnn_lookup):
    model, X, agent_id_to_row = production_model
    sessions = _all_sessions_by_id()
    agent_id = next(aid for aid in agent_id_to_row if aid in sessions)

    payload = build_session_risk_payload(
        agent_id, sessions[agent_id], model, X.iloc[[agent_id_to_row[agent_id]]], gnn_lookup
    )
    jsonschema.validate(payload, SESSION_RISK_PAYLOAD_SCHEMA)
    assert payload["intent_artifact_hash"] == sessions[agent_id].intent_artifact_hash


def test_build_session_risk_payload_hijacked_session_scores_high(production_model, gnn_lookup):
    model, X, agent_id_to_row = production_model
    sessions = _all_sessions_by_id()
    hijacked_id = next(aid for aid, s in sessions.items() if s.injection_present)

    payload = build_session_risk_payload(
        hijacked_id, sessions[hijacked_id], model, X.iloc[[agent_id_to_row[hijacked_id]]], gnn_lookup
    )
    assert payload["risk_level"] in ("MEDIUM", "HIGH")


def test_measure_latency_returns_real_positive_milliseconds(production_model, gnn_lookup):
    result = measure_latency(n_runs=10)
    assert result["n_runs"] == 10
    assert 0 < result["mean_ms"] < 5000  # sanity ceiling, not a tight perf assertion
    assert result["p50_ms"] <= result["p95_ms"] <= result["max_ms"]
