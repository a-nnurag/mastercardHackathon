import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate.merchant_network import build_merchant_network
from defend.gnn import _graph_to_pyg_data


def test_graph_to_pyg_data_shapes_are_consistent():
    graph = build_merchant_network()
    data, merchant_mask, nodes = _graph_to_pyg_data(graph)

    assert data.x.size(0) == len(nodes) == graph.number_of_nodes()
    assert data.x.size(1) == 4  # 3-way one-hot type + registration-age
    assert data.edge_index.size(0) == 2
    assert data.edge_index.size(1) == 2 * graph.number_of_edges()  # both directions
    assert merchant_mask.sum().item() == 218  # 19 legit + 199 fraud-ring


def test_train_and_evaluate_converges_and_returns_real_metrics():
    from defend.gnn import train_and_evaluate

    metrics = train_and_evaluate()
    assert metrics is not None, "GNN did not converge — see defend/gnn.py's fallback path"
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert metrics["f1"] > 0.5  # sanity floor, not a tight bound on the exact number
