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
from defend.gnn import predict_all_merchants, train_and_evaluate
from defend.lightgbm_baseline import build_feature_matrix
from defend.rules import apply_rules
from generate.generated_sessions import load_cached_dataset
from generate.join_ieee_cis import _relabel_benign_subtlety
from mutator.mutate import JOINED_PATH, MUTATION_ROUNDS_PATH, _load_mutator_cache

_DATA_DIR = os.path.dirname(JOINED_PATH)


def load_all_sessions() -> list[dict]:
    """The full 226-session set: 210 from Task 2 + 16 from Task 10's
    Mutator — the same combined lookup mutator/mutate.py itself uses to
    resolve base sessions, reused rather than rebuilt."""
    combined = {d["session"].agent_id: (d["session"], d["subtlety"]) for d in load_cached_dataset()}
    for agent_id, session in _load_mutator_cache().items():
        combined[agent_id] = (session, "subtle")  # every mutated session hardens a subtle case
    sessions = [s for s, _ in combined.values()]
    subtlety_list = _relabel_benign_subtlety([t for _, t in combined.values()])
    return score_sessions(sessions), subtlety_list


def build_session_index() -> tuple[list[dict], dict, dict]:
    """Precomputes every INSTANT signal (no LLM) for all 226 sessions
    once, at startup — rules, constraint_drift, content score, GNN
    probability. Returns (session rows for the UI, lgb_lookup, gnn_lookup)
    so main.py can pass the two lookups straight into
    defend.llm_verdict.verdict_for_session() without recomputing them
    per request."""
    sessions, subtlety_list = load_all_sessions()
    for s in sessions:
        s.constraint_drift = compute_constraint_drift(s)
        s.ingestion_source_trust_score = compute_ingestion_source_trust_score(s)

    gnn_lookup = predict_all_merchants()
    lgb_lookup = _lightgbm_prob_lookup()

    rows = []
    for session, subtlety in zip(sessions, subtlety_list):
        rules_flagged, rules_reasons = apply_rules(session)
        domain = extract_domain(session.task_origin_url)
        content_text = session.injection_payload_text or session.raw_utterance
        rows.append({
            "agent_id": session.agent_id,
            "category": session.mandate_scope.categories[0],
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
                "utterance_artifact_divergence": session.utterance_artifact_divergence,
                "content_score": score_injection_likelihood(content_text),
                "lightgbm_prob": lgb_lookup.get(session.agent_id),
                "gnn_prob": gnn_lookup.get(domain),
            },
        })
    return rows, lgb_lookup, gnn_lookup


def _lightgbm_prob_lookup() -> dict:
    import pandas as pd

    df = pd.read_csv(JOINED_PATH)
    X, y, categorical_cols = build_feature_matrix(df)
    oof = cross_validated_predictions(X, y, df["subtlety"], categorical_cols)
    return dict(zip(df["agent_id"], oof))


def compute_attack1_metrics() -> dict:
    """Task 5's exact methodology (stratified 5-fold CV, out-of-fold),
    re-run once at startup — real numbers, not cached from a prior run
    that could go stale."""
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    df = pd.read_csv(JOINED_PATH)
    X, y, categorical_cols = build_feature_matrix(df)
    oof = cross_validated_predictions(X, y, df["subtlety"], categorical_cols)
    y_pred = (oof >= 0.5).astype(int)

    def _slice(mask):
        mask = pd.Series(mask, index=df.index).values
        yt, yp, ys = y[mask], y_pred[mask], oof[mask]
        if len(set(yt)) < 2:
            return None
        return {
            "n": int(mask.sum()),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
            "auc": roc_auc_score(yt, ys),
        }

    subtlety = df["subtlety"]
    benign_mask = (subtlety == "benign").values
    return {
        "overall": _slice([True] * len(df)),
        "obvious": _slice(subtlety.isin(["obvious", "benign"]).values),
        "subtle": _slice(subtlety.isin(["subtle", "benign"]).values),
        "fp_rate_benign": float(y_pred[benign_mask].mean()) if benign_mask.any() else None,
    }


def compute_attack2_metrics() -> dict | None:
    """Task 8's GNN, re-run once at startup. Returns None (not a fake
    zero-filled dict) if training doesn't converge, matching gnn.py's
    own fallback contract."""
    return train_and_evaluate()


def load_mutation_rounds() -> list[dict]:
    """Task 10's real per-round table. Returns [] — not fabricated
    numbers — if the Mutator hasn't been run in this clone yet."""
    if not os.path.exists(MUTATION_ROUNDS_PATH):
        return []
    with open(MUTATION_ROUNDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
