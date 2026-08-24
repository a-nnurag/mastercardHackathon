import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defend.contracts import (
    SignalResult,
    SignalSet,
    DetectionEvidence,
    RiskDecision,
    RiskExplanation,
)


def test_signal_result_creation_and_to_dict():
    sig = SignalResult(
        name="constraint_drift",
        value=0.5,
        available=True,
        evidence=["amount_cap_breached: signed INR 24,000 > cap INR 8,000"],
        hard_violation=True,
        metadata={"method": "regex_cap_check"},
    )
    assert sig.name == "constraint_drift"
    assert sig.value == 0.5
    assert sig.available is True
    assert sig.hard_violation is True
    assert len(sig.evidence) == 1

    d = sig.to_dict()
    assert d["name"] == "constraint_drift"
    assert d["value"] == 0.5
    assert d["hard_violation"] is True
    assert d["evidence"] == ["amount_cap_breached: signed INR 24,000 > cap INR 8,000"]


def test_signal_set_aggregation():
    sset = SignalSet()
    s1 = SignalResult(name="sig1", value=0.8, available=True, evidence=["ev1"], hard_violation=False)
    s2 = SignalResult(name="sig2", value=None, available=False, evidence=[])
    s3 = SignalResult(name="sig3", value=1.0, available=True, evidence=["ev3"], hard_violation=True)

    sset.add(s1)
    sset.add(s2)
    sset.add(s3)

    assert sset.get("sig1") == s1
    assert sset.get_value("sig1") == 0.8
    assert sset.get_value("sig2", default=0.0) == 0.0  # unavailable returns default
    assert sset.get_value("nonexistent", default=0.5) == 0.5

    assert sset.has_hard_violation() is True
    assert sset.all_evidence() == ["ev1", "ev3"]


def test_risk_decision_validations():
    decision = RiskDecision(
        risk_score=0.85,
        risk_level="HIGH",
        decision="BLOCK",
        signals={"sig1": 0.85},
        evidence=["hard violation observed"],
        attack_family="prompt_injection",
    )
    d = decision.to_dict()
    assert d["risk_score"] == 0.85
    assert d["risk_level"] == "HIGH"
    assert d["decision"] == "BLOCK"
    assert d["attack_family"] == "prompt_injection"

    # Invalid score
    with pytest.raises(ValueError, match="risk_score"):
        RiskDecision(risk_score=1.5, risk_level="HIGH", decision="BLOCK")

    # Invalid risk level
    with pytest.raises(ValueError, match="risk_level"):
        RiskDecision(risk_score=0.5, risk_level="CRITICAL", decision="BLOCK")

    # Invalid decision
    with pytest.raises(ValueError, match="decision"):
        RiskDecision(risk_score=0.5, risk_level="HIGH", decision="DENY")


def test_risk_explanation():
    explanation = RiskExplanation(
        summary="Transaction blocked due to exceeding mandate amount cap.",
        evidence_summary=["Signed INR 24,000 exceeds allowed INR 8,000."],
    )
    d = explanation.to_dict()
    assert d["summary"] == "Transaction blocked due to exceeding mandate amount cap."
    assert len(d["evidence_summary"]) == 1
