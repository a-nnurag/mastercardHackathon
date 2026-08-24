import sys
import os
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defend.contracts import RiskDecision, RiskExplanation, SignalResult, SignalSet
from defend.detection_context import DetectionContext
from defend.gnn import MerchantGNN, predict_all_merchants, predict_merchants
from defend.integration_contract import _get_production_models_and_features, build_session_risk_payload
from defend.lightgbm_baseline import LightGBMDetector, predict_probability
from defend.model_registry import ModelRegistry
from defend.risk_engine import RiskEngine
from generate.merchant_network import build_merchant_network
from generate.session_schema import AgentSession, MandateScope


def test_gnn_predict_merchants_does_not_train():
    model = MerchantGNN(in_dim=4)
    model.eval()

    # Record weights before inference
    weight_before = model.conv1.lin_l.weight.clone()

    graph = build_merchant_network()
    from defend.gnn import _graph_to_pyg_data
    data, merchant_mask, nodes = _graph_to_pyg_data(graph)

    preds = predict_merchants(model, data, merchant_mask, nodes)
    assert len(preds) > 0

    # Assert weights have not changed at all
    weight_after = model.conv1.lin_l.weight
    assert (weight_before == weight_after).all()


def test_risk_engine_evaluate_is_pure_computation():
    engine = RiskEngine()
    signal_set = SignalSet()
    signal_set.add(SignalResult(name="constraint_drift", value=0.5, available=True))
    signal_set.add(SignalResult(name="lightgbm_prob", value=0.8, available=True))

    start = time.perf_counter()
    decision = engine.evaluate(signal_set)
    duration_ms = (time.perf_counter() - start) * 1000

    assert isinstance(decision, RiskDecision)
    assert decision.decision == "BLOCK"
    assert duration_ms < 50.0  # Pure computation should be well under 50ms


def test_model_registry_does_not_train_on_cache_hit():
    ModelRegistry.clear_cache()
    m1 = ModelRegistry.get_gnn()
    m2 = ModelRegistry.get_gnn()
    assert m1 is m2  # Exact same cached instance
