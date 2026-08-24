import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defend.contracts import RiskDecision, RiskExplanation
from defend.gnn import MerchantGNN, predict_all_merchants
from defend.llm_verdict import consistency_check, synthesize_explanation, synthesize_verdict
from generate.llm_adapter import MockAdapter
from generate.merchant_network import build_merchant_network


def test_consistency_check_all_clear_expects_low():
    verdict = {"risk_level": "LOW"}
    assert consistency_check(verdict, rules_flagged=False, lightgbm_prob=0.05, content_score=0.02, gnn_prob=0.1)


def test_consistency_check_all_clear_rejects_high():
    verdict = {"risk_level": "HIGH"}
    assert not consistency_check(verdict, rules_flagged=False, lightgbm_prob=0.05, content_score=0.02, gnn_prob=0.1)


def test_consistency_check_multiple_high_signals_expects_medium_or_high():
    verdict = {"risk_level": "HIGH"}
    assert consistency_check(verdict, rules_flagged=True, lightgbm_prob=0.95, content_score=0.4, gnn_prob=0.9)

    verdict_low = {"risk_level": "LOW"}
    assert not consistency_check(verdict_low, rules_flagged=True, lightgbm_prob=0.95, content_score=0.4, gnn_prob=0.9)


def test_consistency_check_one_weak_signal_is_lenient():
    assert consistency_check({"risk_level": "LOW"}, rules_flagged=False, lightgbm_prob=0.6, content_score=0.02, gnn_prob=0.1)
    assert consistency_check({"risk_level": "MEDIUM"}, rules_flagged=False, lightgbm_prob=0.6, content_score=0.02, gnn_prob=0.1)


def test_synthesize_explanation_returns_typed_explanation():
    decision = RiskDecision(
        risk_score=0.88,
        risk_level="HIGH",
        decision="BLOCK",
        signals={"lightgbm_prob": 0.88},
        evidence=["high probability prompt injection"],
    )
    session_summary = {
        "raw_utterance": "Book flight",
        "signed_artifact_text": "Buy electronics",
        "domain": "fake-store.com",
    }
    mock_adapter = MockAdapter({
        "summary": "Transaction blocked due to divergence between request and signed artifact.",
        "evidence_summary": ["High model score indicating prompt injection."],
    })

    explanation = synthesize_explanation(decision, session_summary, adapter=mock_adapter)
    assert isinstance(explanation, RiskExplanation)
    assert "blocked" in explanation.summary.lower()
    assert len(explanation.evidence_summary) > 0


def test_llm_cannot_override_risk_decision():
    # Even if LLM returns a lenient description, the finalized decision remains BLOCK
    rules_result = (True, ["amount cap breached"])
    session_summary = {
        "raw_utterance": "Book flight",
        "signed_artifact_text": "Buy electronics 400000 INR",
        "domain": "fake-store.com",
    }
    # Mock LLM returns mild text
    mock_adapter = MockAdapter({
        "summary": "Everything looks fine to me.",
        "evidence_summary": ["No issues seen."],
    })

    verdict = synthesize_verdict(
        session_summary, rules_result, lightgbm_prob=0.05, content_score=0.0, gnn_prob=0.0, adapter=mock_adapter
    )
    # The authority is RiskEngine, so decision MUST be BLOCK despite mock LLM text
    assert verdict["decision"] == "BLOCK"
    assert verdict["risk_level"] == "HIGH"
    assert verdict["recommendation"] == "BLOCK"


def test_predict_all_merchants_covers_every_real_fraud_ring_domain():
    graph = build_merchant_network()
    fraud_domains = {n for n, d in graph.nodes(data=True) if d.get("type") == "merchant" and d["is_fraud_ring_member"]}

    model = MerchantGNN(in_dim=4)
    model.eval()
    predictions = predict_all_merchants(model, graph)
    assert fraud_domains <= predictions.keys()
    for prob in predictions.values():
        assert 0.0 <= prob <= 1.0
