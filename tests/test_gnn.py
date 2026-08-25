import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from generate.merchant_network import build_merchant_network
from defend.gnn import _graph_to_pyg_data, MerchantGNN, predict_all_merchants, train_and_evaluate


def test_graph_to_pyg_data_shapes_are_consistent():
    graph = build_merchant_network()
    data, merchant_mask, nodes = _graph_to_pyg_data(graph)

    assert data.x.size(0) == len(nodes) == graph.number_of_nodes()
    assert data.x.size(1) == 4  # 3-way one-hot type + registration-age
    assert data.edge_index.size(0) == 2
    assert data.edge_index.size(1) == 2 * graph.number_of_edges()  # both directions
    assert merchant_mask.sum().item() > 0


def test_predict_all_merchants_requires_model():
    with pytest.raises(ValueError, match="requires a trained MerchantGNN"):
        predict_all_merchants(None)


def test_predict_all_merchants_pure_inference():
    from defend.model_registry import ModelRegistry
    model = ModelRegistry.get_gnn()
    if model is None:
        model = MerchantGNN(in_dim=4)
        model.eval()
    predictions = predict_all_merchants(model)
    assert len(predictions) > 0
    for prob in predictions.values():
        assert 0.0 <= prob <= 1.0


def test_train_and_evaluate_converges_and_returns_real_metrics():
    metrics = train_and_evaluate()
    assert metrics is not None, "GNN did not converge"
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
