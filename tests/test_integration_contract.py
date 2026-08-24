import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jsonschema
import pytest

from defend.integration_contract import (
    SESSION_RISK_PAYLOAD_SCHEMA,
    _all_sessions_by_id,
    _get_production_models_and_features,
    build_session_risk_payload,
    measure_latency,
)
from defend.gnn import predict_all_merchants
from defend.model_registry import ModelRegistry
from defend.risk_engine import RiskEngine, RiskFusionPolicy


def test_risk_policy_rules_flag_forces_block():
    engine = RiskEngine()
    from defend.contracts import SignalResult, SignalSet
    sset = SignalSet()
    sset.add(SignalResult(name="rules", value=True, available=True, hard_violation=True, evidence=["rule breach"]))
    decision = engine.evaluate(sset)
    assert decision.decision == "BLOCK"
    assert decision.risk_level == "HIGH"


@pytest.fixture(scope="module")
def production_setup():
    return _get_production_models_and_features()


def test_build_session_risk_payload_matches_json_schema(production_setup):
    lgb_model, gnn_model, X, agent_id_to_row, gnn_lookup = production_setup
    sessions = _all_sessions_by_id()
    agent_id = next(aid for aid in agent_id_to_row if aid in sessions)

    payload = build_session_risk_payload(
        agent_id, sessions[agent_id], lgb_model, X.iloc[[agent_id_to_row[agent_id]]], gnn_lookup
    )
    jsonschema.validate(payload, SESSION_RISK_PAYLOAD_SCHEMA)
    assert payload["intent_artifact_hash"] == sessions[agent_id].intent_artifact_hash
    assert payload["decision"] in ("ALLOW", "HOLD", "BLOCK")


def test_build_session_risk_payload_hijacked_session_scores_high(production_setup):
    lgb_model, gnn_model, X, agent_id_to_row, gnn_lookup = production_setup
    sessions = _all_sessions_by_id()
    hijacked_id = next(aid for aid, s in sessions.items() if s.injection_present)

    payload = build_session_risk_payload(
        hijacked_id, sessions[hijacked_id], lgb_model, X.iloc[[agent_id_to_row[hijacked_id]]], gnn_lookup
    )
    assert payload["risk_level"] in ("MEDIUM", "HIGH")
    assert payload["decision"] in ("HOLD", "BLOCK")


def test_measure_latency_returns_real_positive_milliseconds(production_setup):
    result = measure_latency(n_runs=10)
    assert result["n_runs"] == 10
    assert 0 < result["mean_ms"] < 5000  # sanity ceiling
    assert result["p50_ms"] <= result["p95_ms"] <= result["max_ms"]
