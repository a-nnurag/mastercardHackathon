import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defend.llm_verdict import consistency_check
from defend.gnn import predict_all_merchants
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
    # A single borderline signal doesn't force a specific risk_level either way
    assert consistency_check({"risk_level": "LOW"}, rules_flagged=False, lightgbm_prob=0.6, content_score=0.02, gnn_prob=0.1)
    assert consistency_check({"risk_level": "MEDIUM"}, rules_flagged=False, lightgbm_prob=0.6, content_score=0.02, gnn_prob=0.1)


def test_predict_all_merchants_covers_every_real_fraud_ring_domain():
    # No live LLM call -- this exercises the GNN training path only.
    graph = build_merchant_network()
    fraud_domains = {n for n, d in graph.nodes(data=True) if d.get("type") == "merchant" and d["is_fraud_ring_member"]}

    predictions = predict_all_merchants()
    assert fraud_domains <= predictions.keys()
    for prob in predictions.values():
        assert 0.0 <= prob <= 1.0
