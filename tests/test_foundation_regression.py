import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from defend.contracts import RiskDecision, SignalResult, SignalSet
from defend.constraint_drift import compute_constraint_drift, compute_ingestion_source_trust_score
from defend.content_layer import score_injection_likelihood
from defend.gnn import MerchantGNN, predict_all_merchants
from defend.lightgbm_baseline import build_feature_matrix, train_baseline, predict_probability
from defend.risk_engine import RiskEngine
from defend.rules import apply_rules
from generate.merchant_network import build_merchant_network
from generate.session_schema import AgentSession, MandateScope
from generate.synthetic_sessions import LEGITIMATE, HIJACKED


def test_attack1_prompt_injection_detection_regression():
    # Verify all hijacked sessions in synthetic baseline produce strong risk signals
    for session in HIJACKED:
        drift = compute_constraint_drift(session)
        trust = compute_ingestion_source_trust_score(session)
        rules_flagged, rules_reasons = apply_rules(session)

        # Either rule breached, or positive constraint drift
        assert rules_flagged or drift > 0.0 or trust > 0.0

        # Construct SignalSet and evaluate RiskEngine
        sset = SignalSet()
        sset.add(SignalResult(name="rules", value=rules_flagged, available=True, evidence=rules_reasons, hard_violation=rules_flagged))
        sset.add(SignalResult(name="constraint_drift", value=drift, available=True))
        sset.add(SignalResult(name="ingestion_source_trust_score", value=trust, available=True))

        engine = RiskEngine()
        decision = engine.evaluate(sset)
        assert decision.decision in ("HOLD", "BLOCK")
        assert decision.risk_level in ("MEDIUM", "HIGH")


def test_benign_sessions_do_not_produce_false_high_risk_regression():
    # Verify benign sessions do not falsely trigger high risk
    engine = RiskEngine()
    for session in LEGITIMATE:
        rules_flagged, rules_reasons = apply_rules(session)
        drift = compute_constraint_drift(session)
        trust = compute_ingestion_source_trust_score(session)

        assert not rules_flagged
        assert drift == 0.0
        assert trust == 0.0

        sset = SignalSet()
        sset.add(SignalResult(name="rules", value=rules_flagged, available=True, evidence=rules_reasons, hard_violation=rules_flagged))
        sset.add(SignalResult(name="constraint_drift", value=drift, available=True))
        sset.add(SignalResult(name="ingestion_source_trust_score", value=trust, available=True))

        decision = engine.evaluate(sset)
        assert decision.decision == "ALLOW"
        assert decision.risk_level == "LOW"


def test_attack2_merchant_laundering_regression():
    graph = build_merchant_network()
    model = MerchantGNN(in_dim=4)
    model.eval()

    predictions = predict_all_merchants(model, graph)
    assert len(predictions) > 0

    # Ensure all merchants in graph receive valid [0, 1] probability
    for domain, prob in predictions.items():
        assert 0.0 <= prob <= 1.0
