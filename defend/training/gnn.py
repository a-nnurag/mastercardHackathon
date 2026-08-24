"""
Offline training script for Merchant GNN model.
"""

import os
from defend.gnn import _graph_to_pyg_data, train_gnn, save_gnn_model
from generate.merchant_network import build_merchant_network

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "artifacts", "models", "gnn.pt")


def train_and_save_gnn(model_save_path: str = DEFAULT_MODEL_PATH):
    print("[GNN] Building merchant network graph...")
    graph = build_merchant_network()
    data, merchant_mask, nodes = _graph_to_pyg_data(graph)
    print(f"[GNN] PyG Graph: {data.num_nodes} nodes, {data.num_edges} edges, {int(merchant_mask.sum())} merchant nodes")

    print("[GNN] Training MerchantGNN model...")
    model, losses = train_gnn(data, merchant_mask)
    save_gnn_model(model, model_save_path)
    print(f"[GNN] Model weights saved to {model_save_path} (Final loss: {losses[-1]:.4f})")
    return model


if __name__ == "__main__":
    train_and_save_gnn()
