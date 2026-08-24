"""
GNN over the merchant/acquirer/beneficiary graph — Defend layer 4 per
TEAM_BRIEF.md Sec 4.6, attack #2 (transaction laundering via fraudulent
merchant network).

Architectural Invariants:
- Online inference is strictly separated from offline training.
- predict_merchants / predict_all_merchants requires a trained MerchantGNN model.
- Inference NEVER runs optimizer steps, loss calculation, or backward passes.
"""

import os
from typing import Optional

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

# Ensure single-threaded CPU execution on macOS Python 3.13 to prevent autograd threadpool crashes
torch.set_num_threads(1)
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from defend.constraint_drift import extract_domain
from defend.contracts import SignalResult
from defend.detection_context import DetectionContext
from defend.signals.base import SignalDetector
from generate.merchant_network import build_merchant_network
from generate.session_schema import AgentSession

_SEED = 42
_TEST_SIZE = 0.3
_EPOCHS = 100
_HIDDEN_DIM = 16


class MerchantGNN(torch.nn.Module):
    def __init__(self, in_dim: int = 4, hidden_dim: int = _HIDDEN_DIM):
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


def train_gnn(
    data: Data,
    merchant_mask: torch.Tensor,
    train_mask: Optional[torch.Tensor] = None,
    epochs: int = _EPOCHS,
    seed: int = _SEED,
    lr: float = 0.01,
) -> tuple[MerchantGNN, list[float]]:
    """Pure offline training function for MerchantGNN."""
    torch.manual_seed(seed)
    model = MerchantGNN(in_dim=data.x.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    mask = train_mask if train_mask is not None else merchant_mask

    losses = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(logits[mask], data.y[mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    return model, losses


def save_gnn_model(model: MerchantGNN, path: str) -> None:
    """Persists trained GNN weights to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_gnn_model(path: str, in_dim: int = 4, hidden_dim: int = _HIDDEN_DIM) -> MerchantGNN:
    """Loads trained GNN model weights from disk in evaluation mode."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"GNN model artifact not found at {path}")
    model = MerchantGNN(in_dim=in_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict_merchants(
    model: MerchantGNN,
    data: Data,
    merchant_mask: torch.Tensor,
    nodes: list[str],
) -> dict[str, float]:
    """Pure online inference function — computes fraud ring probabilities without training."""
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(data.x, data.edge_index)).numpy()

    merchant_indices = merchant_mask.nonzero(as_tuple=True)[0].tolist()
    return {nodes[i]: float(probs[i]) for i in merchant_indices}


def predict_all_merchants(model: MerchantGNN, graph: Optional[nx.Graph] = None) -> dict[str, float]:
    """
    Computes {domain: fraud_probability} for all merchant nodes using a PRE-TRAINED model.
    Enforces strict architectural invariant: model must be provided; inference never trains.
    """
    if model is None:
        raise ValueError("predict_all_merchants requires a trained MerchantGNN model instance.")
    if graph is None:
        graph = build_merchant_network()
    data, merchant_mask, nodes = _graph_to_pyg_data(graph)
    return predict_merchants(model, data, merchant_mask, nodes)


def train_and_evaluate(graph: Optional[nx.Graph] = None) -> Optional[dict]:
    """Offline evaluation function (Task 8 held-out evaluation)."""
    if graph is None:
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

    model, losses = train_gnn(data, merchant_mask, train_mask=train_mask, epochs=_EPOCHS, seed=_SEED)

    converged = losses[-1] < losses[0] * 0.5 and not any(np.isnan(losses))
    print(f"Loss: epoch 0 = {losses[0]:.4f}, epoch {_EPOCHS - 1} = {losses[-1]:.4f} "
          f"({'converged' if converged else 'DID NOT CONVERGE'})")

    if not converged:
        print("GNN training did not converge within reasonable effort.")
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


class GNNDetector(SignalDetector):
    name: str = "gnn_prob"
    supported_attack_families: tuple[str, ...] = ("merchant_laundering",)

    def __init__(self, gnn_lookup: Optional[dict[str, float]] = None):
        self.gnn_lookup = gnn_lookup

    def detect(
        self,
        session: AgentSession,
        context: Optional[DetectionContext] = None,
    ) -> SignalResult:
        domain = extract_domain(session.task_origin_url)
        lookup = self.gnn_lookup
        if lookup is None and context is not None:
            if "gnn_lookup" in context.metadata:
                lookup = context.metadata["gnn_lookup"]
            else:
                model = context.get_model("gnn")
                if model is not None:
                    lookup = predict_all_merchants(model, context.graph)

        if lookup is None:
            return SignalResult(
                name=self.name,
                value=None,
                available=False,
                evidence=[],
                hard_violation=False,
                metadata={"reason": "gnn_model_or_lookup_unavailable"},
            )

        prob = lookup.get(domain)
        if prob is None:
            return SignalResult(
                name=self.name,
                value=None,
                available=False,
                evidence=[],
                hard_violation=False,
                metadata={"reason": "destination_not_in_merchant_network", "domain": domain},
            )

        evidence = []
        if prob >= 0.5:
            evidence.append(f"merchant_network_risk_high: domain '{domain}' fraud ring probability {prob:.3f} >= 0.50")
        elif prob >= 0.15:
            evidence.append(f"merchant_network_risk_medium: domain '{domain}' fraud ring probability {prob:.3f} >= 0.15")

        return SignalResult(
            name=self.name,
            value=prob,
            available=True,
            evidence=evidence,
            hard_violation=False,
            metadata={
                "domain": domain,
                "network_risk_probability": prob,
                "signal_concept": "network_relationship_intelligence",
            },
        )


if __name__ == "__main__":
    train_and_evaluate()
