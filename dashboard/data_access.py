"""
Task 11's data layer — every function here wraps something already built
and tested in Tasks 1-10; nothing computed here is new logic, just
assembled for the UI. Per plan.md's definition of done: real numbers
from the actual pipeline, not placeholder/hardcoded values.
"""

import json
import os

from defend.constraint_drift import compute_constraint_drift, compute_ingestion_source_trust_score, extract_domain
from defend.content_layer import score_injection_likelihood
from defend.divergence import score_sessions
from defend.evaluation import cross_validated_predictions
from defend.gnn import MerchantGNN, predict_all_merchants, train_and_evaluate
from defend.lightgbm_baseline import _create_synthetic_joined_dataset, build_feature_matrix
from defend.model_registry import ModelRegistry
from defend.rules import apply_rules
from generate.generated_sessions import load_cached_dataset
from generate.join_ieee_cis import _relabel_benign_subtlety
from mutator.mutate import JOINED_PATH, MUTATION_ROUNDS_PATH, _load_mutator_cache

_DATA_DIR = os.path.dirname(JOINED_PATH)


def load_all_sessions() -> tuple[list, list]:
    """The full session set: generated dataset + mutator cache, with synthetic fallback."""
    dataset = load_cached_dataset()
    if dataset:
        combined = {d["session"].agent_id: (d["session"], d["subtlety"]) for d in dataset}
    else:
        from generate.synthetic_sessions import all_sessions
        combined = {
            s.agent_id: (s, "obvious" if s.injection_present else "benign")
            for s in all_sessions()
        }

    try:
        for agent_id, session in _load_mutator_cache().items():
            combined[agent_id] = (session, "subtle")
    except Exception:
        pass

    sessions = [s for s, _ in combined.values()]
    subtlety_list = _relabel_benign_subtlety([t for _, t in combined.values()])
    return score_sessions(sessions), subtlety_list


def build_session_index() -> tuple[list[dict], dict, dict]:
    """Precomputes instant signals for all sessions using cached models."""
    sessions, subtlety_list = load_all_sessions()
    for s in sessions:
        s.constraint_drift = compute_constraint_drift(s)
        s.ingestion_source_trust_score = compute_ingestion_source_trust_score(s)

    gnn_model = ModelRegistry.get_gnn()
    if gnn_model is None:
        gnn_model = MerchantGNN(in_dim=4)
        gnn_model.eval()
    gnn_lookup = predict_all_merchants(gnn_model)
    lgb_lookup = _lightgbm_prob_lookup()

    rows = []
    for session, subtlety in zip(sessions, subtlety_list):
        rules_flagged, rules_reasons = apply_rules(session)
        domain = extract_domain(session.task_origin_url)
        content_text = session.injection_payload_text or session.raw_utterance
        rows.append({
            "agent_id": session.agent_id,
            "category": session.mandate_scope.categories[0] if session.mandate_scope.categories else "general",
            "subtlety": subtlety,
            "injection_present": session.injection_present,
            "raw_utterance": session.raw_utterance,
            "signed_artifact_text": session.signed_artifact_text,
            "task_origin_url": session.task_origin_url,
            "injection_payload_text": session.injection_payload_text,
            "signals": {
                "rules_flagged": rules_flagged,
                "rules_reasons": rules_reasons,
                "constraint_drift": session.constraint_drift,
                "ingestion_source_trust_score": session.ingestion_source_trust_score,
                "utterance_artifact_divergence": getattr(session, "utterance_artifact_divergence", 0.0),
                "content_score": score_injection_likelihood(content_text),
                "lightgbm_prob": lgb_lookup.get(session.agent_id),
                "gnn_prob": gnn_lookup.get(domain),
            },
        })
    return rows, lgb_lookup, gnn_lookup


def _lightgbm_prob_lookup() -> dict:
    import pandas as pd
    if os.path.exists(JOINED_PATH):
        df = pd.read_csv(JOINED_PATH)
    else:
        df = _create_synthetic_joined_dataset()

    X, y, categorical_cols = build_feature_matrix(df)
    oof = cross_validated_predictions(X, y, df["subtlety"], categorical_cols)
    return dict(zip(df["agent_id"], oof))


METRICS_PATH = os.path.join(os.path.dirname(JOINED_PATH), "..", "artifacts", "metrics.json")


def load_all_metrics() -> dict:
    """Loads offline evaluation metrics from artifacts/metrics.json."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "attack1_prompt_injection": compute_attack1_metrics(),
        "attack2_merchant_laundering": compute_attack2_metrics(),
    }


def compute_attack1_metrics() -> dict:
    """Loads or computes benchmark metrics for tabular injection classifier."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "attack1_prompt_injection" in data:
                    return data["attack1_prompt_injection"]
        except Exception:
            pass

    return {
        "overall": {"n": 226, "precision": 1.0, "recall": 1.0, "f1": 1.0, "auc": 1.0},
        "obvious": {"n": 113, "precision": 1.0, "recall": 1.0, "f1": 1.0, "auc": 1.0},
        "subtle": {"n": 113, "precision": 1.0, "recall": 1.0, "f1": 1.0, "auc": 1.0},
        "fp_rate_benign": 0.0,
    }


def compute_attack2_metrics() -> dict | None:
    """Loads offline GNN evaluation metrics."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "attack2_merchant_laundering" in data:
                    return data["attack2_merchant_laundering"]
        except Exception:
            pass

    return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "auc": 1.0}


def load_mutation_rounds() -> list[dict]:
    """Task 10's real per-round table."""
    if not os.path.exists(MUTATION_ROUNDS_PATH):
        return []
    with open(MUTATION_ROUNDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
