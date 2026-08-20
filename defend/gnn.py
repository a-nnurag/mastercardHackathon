"""
GNN over the merchant/acquirer/beneficiary graph — Defend layer 4 per
TEAM_BRIEF.md Sec 4.6, attack #2 (transaction laundering via fraudulent
merchant network). Optional, has a named fallback per plan.md Task 8: if
this doesn't converge with reasonable effort, drop to graph features
(degree, community ID) fed into LightGBM instead — see
defend/gnn_fallback.py, only written if this file actually needed it.

Deliberate simplification: converts the networkx graph (3 node types:
merchant/acquirer/beneficiary) into a HOMOGENEOUS PyG graph (one-hot type
feature + registration-age feature per node) rather than a strict typed
HeteroData. Message passing still runs over the real
merchant-acquirer-beneficiary edges — a merchant's embedding still
aggregates its acquirer/beneficiary neighbors — it's just not
implemented with per-edge-type convolutions. Chosen for reliability: a
plain 2-layer GraphSAGE is far less likely to hit PyG API friction than
HeteroConv, and plan.md explicitly treats instability as a real possible
outcome here, not a failure to avoid at all costs.

Transductive node classification (standard for a single-graph GNN):
train/test split is over node LABELS only, not separate graphs — the
full graph structure is visible during training, only some merchant
nodes' fraud-ring labels are held out for evaluation.
"""

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from generate.merchant_network import build_merchant_network

_SEED = 42
_TEST_SIZE = 0.3
_EPOCHS = 100
_HIDDEN_DIM = 16


class MerchantGNN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = _HIDDEN_DIM):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index).squeeze(-1)  # one logit per node


def _graph_to_pyg_data(graph: nx.Graph):
    nodes = list(graph.nodes())
    index_of = {n: i for i, n in enumerate(nodes)}
    type_to_onehot = {"merchant": [1, 0, 0], "acquirer": [0, 1, 0], "beneficiary": [0, 0, 1]}

    max_age = max((d.get("registered_days_ago", 0) for _, d in graph.nodes(data=True)), default=1)

    features, labels, merchant_mask = [], [], []
    for n in nodes:
        attrs = graph.nodes[n]
        onehot = type_to_onehot[attrs["type"]]
        age_norm = attrs.get("registered_days_ago", 0) / max_age
        features.append(onehot + [age_norm])

        if attrs["type"] == "merchant":
            labels.append(int(attrs["is_fraud_ring_member"]))
            merchant_mask.append(True)
        else:
            labels.append(-1)
            merchant_mask.append(False)

    edges = []
    for u, v in graph.edges():
        edges.append((index_of[u], index_of[v]))
        edges.append((index_of[v], index_of[u]))

    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor(labels, dtype=torch.float)
    merchant_mask = torch.tensor(merchant_mask, dtype=torch.bool)

    return Data(x=x, edge_index=edge_index, y=y), merchant_mask, nodes


def train_and_evaluate():
    graph = build_merchant_network()
    data, merchant_mask, nodes = _graph_to_pyg_data(graph)

    merchant_indices = merchant_mask.nonzero(as_tuple=True)[0].tolist()
    merchant_labels = data.y[merchant_mask].tolist()
    train_idx, test_idx = train_test_split(
        merchant_indices, test_size=_TEST_SIZE, random_state=_SEED, stratify=merchant_labels
    )
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True

    torch.manual_seed(_SEED)
    model = MerchantGNN(in_dim=data.x.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    losses = []
    model.train()
    for epoch in range(_EPOCHS):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    converged = losses[-1] < losses[0] * 0.5 and not any(np.isnan(losses))
    print(f"Loss: epoch 0 = {losses[0]:.4f}, epoch {_EPOCHS - 1} = {losses[-1]:.4f} "
          f"({'converged' if converged else 'DID NOT CONVERGE'})")

    if not converged:
        print("GNN training did not converge within reasonable effort — falling back per plan.md Task 8.")
        return None

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(data.x, data.edge_index))[test_mask].numpy()
    y_true = data.y[test_mask].numpy()
    y_pred = (probs >= 0.5).astype(int)

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, probs) if len(set(y_true)) > 1 else float("nan"),
    }
    print(f"\nHeld-out merchant nodes (n={len(y_true)}): "
          f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
          f"f1={metrics['f1']:.3f} auc={metrics['auc']:.3f}")
    return metrics


def predict_all_merchants() -> dict[str, float]:
    """Trains on ALL merchant labels (not held out) and returns
    {domain: fraud_probability} for every merchant node. For Task 9's
    operational verdict scoring, not an accuracy claim — that's
    train_and_evaluate()'s job, kept separate on purpose so this function
    is never mistaken for a held-out result."""
    graph = build_merchant_network()
    data, merchant_mask, nodes = _graph_to_pyg_data(graph)

    torch.manual_seed(_SEED)
    model = MerchantGNN(in_dim=data.x.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for _ in range(_EPOCHS):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(logits[merchant_mask], data.y[merchant_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(data.x, data.edge_index)).numpy()

    merchant_indices = merchant_mask.nonzero(as_tuple=True)[0].tolist()
    return {nodes[i]: float(probs[i]) for i in merchant_indices}


if __name__ == "__main__":
    train_and_evaluate()
