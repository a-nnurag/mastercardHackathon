import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defend.contracts import SignalResult, SignalSet, RiskDecision
from defend.risk_engine import RiskEngine, RiskFusionPolicy


def test_invariant_1_no_signals_safe_fallback():
    engine = RiskEngine()
    empty_set = SignalSet()
    decision = engine.evaluate(empty_set)

    assert isinstance(decision, RiskDecision)
    # Fail-closed default policy on completely missing signals
    assert decision.decision == "HOLD"
    assert decision.risk_level == "MEDIUM"
    assert len(decision.evidence) > 0

    # Configurable fail-open if explicitly desired
    allow_policy = RiskFusionPolicy(empty_signals_action="ALLOW")
    allow_decision = engine.evaluate(empty_set, policy=allow_policy)
    assert allow_decision.decision == "ALLOW"
    assert allow_decision.risk_level == "LOW"


def test_invariant_2_single_signal():
    engine = RiskEngine()
    sset = SignalSet()
    sset.add(SignalResult(name="constraint_drift", value=0.5, available=True, evidence=["drift detected"]))
    decision = engine.evaluate(sset)

    assert decision.decision == "BLOCK"
    assert decision.risk_level == "HIGH"
    assert decision.risk_score >= 0.5
    assert "drift detected" in decision.evidence


def test_invariant_3_multiple_signals_fusion():
    engine = RiskEngine()
    sset = SignalSet()
    sset.add(SignalResult(name="constraint_drift", value=0.0, available=True))
    sset.add(SignalResult(name="lightgbm_prob", value=0.05, available=True))
    sset.add(SignalResult(name="content_injection", value=0.02, available=True))
    decision = engine.evaluate(sset)

    assert decision.decision == "ALLOW"
    assert decision.risk_level == "LOW"
    assert decision.risk_score < 0.15


def test_invariant_4_missing_gnn_resilience():
    engine = RiskEngine()
    sset = SignalSet()
    # GNN is missing / unavailable
    sset.add(SignalResult(name="gnn_prob", value=None, available=False))
    sset.add(SignalResult(name="lightgbm_prob", value=0.85, available=True, evidence=["high lgb hijack probability"]))
    decision = engine.evaluate(sset)

    assert decision.decision == "BLOCK"
    assert decision.risk_level == "HIGH"
    assert decision.risk_score >= 0.85


def test_invariant_5_hard_rule_violation_cannot_be_allow():
    engine = RiskEngine()
    sset = SignalSet()
    # All continuous signals low, but hard rule violation exists
    sset.add(SignalResult(name="lightgbm_prob", value=0.01, available=True))
    sset.add(SignalResult(name="rules", value=True, available=True, hard_violation=True, evidence=["mandate_amount_exceeded"]))
    decision = engine.evaluate(sset)

    assert decision.decision != "ALLOW"
    assert decision.decision == "BLOCK"
    assert decision.risk_level == "HIGH"
    assert decision.risk_score >= 0.85
    assert "mandate_amount_exceeded" in decision.evidence


def test_invariant_6_score_bounded_zero_to_one():
    engine = RiskEngine()
    sset = SignalSet()
    sset.add(SignalResult(name="constraint_drift", value=1.0, available=True))
    sset.add(SignalResult(name="lightgbm_prob", value=1.0, available=True))
    decision = engine.evaluate(sset)

    assert 0.0 <= decision.risk_score <= 1.0


def test_invariant_7_risk_level_consistency():
    engine = RiskEngine()
    policy = RiskFusionPolicy(high_threshold=0.50, medium_threshold=0.15)

    # Low
    s_low = SignalSet({
        "constraint_drift": SignalResult(name="constraint_drift", value=0.0, available=True),
        "lightgbm_prob": SignalResult(name="lightgbm_prob", value=0.05, available=True),
    })
    d_low = engine.evaluate(s_low, policy)
    assert d_low.risk_level == "LOW"
    assert d_low.decision == "ALLOW"

    # Medium
    s_med = SignalSet({
        "content_injection": SignalResult(name="content_injection", value=0.20, available=True),
        "lightgbm_prob": SignalResult(name="lightgbm_prob", value=0.25, available=True),
    })
    d_med = engine.evaluate(s_med, policy)
    assert d_med.risk_level == "MEDIUM"
    assert d_med.decision == "HOLD"

    # High
    s_high = SignalSet({
        "lightgbm_prob": SignalResult(name="lightgbm_prob", value=0.75, available=True),
    })
    d_high = engine.evaluate(s_high, policy)
    assert d_high.risk_level == "HIGH"
    assert d_high.decision == "BLOCK"


def test_invariant_8_attack_family_preservation():
    engine = RiskEngine()

    # Attack 2 GNN laundering
    s_gnn = SignalSet({
        "gnn_prob": SignalResult(name="gnn_prob", value=0.88, available=True, evidence=["shell merchant ring"]),
    })
    d_gnn = engine.evaluate(s_gnn)
    assert d_gnn.attack_family == "merchant_laundering"

    # Attack 1 Prompt Injection
    s_pi = SignalSet({
        "constraint_drift": SignalResult(name="constraint_drift", value=0.5, available=True),
        "content_injection": SignalResult(name="content_injection", value=0.3, available=True),
    })
    d_pi = engine.evaluate(s_pi)
    assert d_pi.attack_family == "prompt_injection"


def test_invariant_9_evidence_survives_fusion():
    engine = RiskEngine()
    sset = SignalSet()
    sset.add(SignalResult(name="s1", value=0.3, available=True, evidence=["evidence_item_1", "evidence_item_2"]))
    sset.add(SignalResult(name="s2", value=0.7, available=True, evidence=["evidence_item_3"]))

    decision = engine.evaluate(sset)
    assert "evidence_item_1" in decision.evidence
    assert "evidence_item_2" in decision.evidence
    assert "evidence_item_3" in decision.evidence
