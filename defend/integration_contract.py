"""
Task 12 -- integration contract + latency measurement.

Per TEAM_BRIEF.md Sec 3.3: this system can't run at the Mastercard
network layer, because `utterance_artifact_divergence`/`constraint_drift`
need the raw human utterance, which the network never sees. It runs at
the agent-provider/PSP layer instead, where the utterance lives, and
emits a session-risk score that rides along with the signed Intent
Artifact into the network -- Decision Intelligence consumes that score,
it doesn't compute it (TEAM_BRIEF.md Part 9 Q&A).

This module builds that payload from the four fast, deterministic/ML
Defend layers only (rules, LightGBM, content, GNN) -- deliberately
excluding Task 9's LLM narrative verdict (defend/llm_verdict.py), which
stays an on-demand, human-facing explanation layer for a risk analyst,
not something that belongs in a per-transaction latency budget. The
LightGBM model used here is fit once on all 226 sessions
(defend/lightgbm_baseline.py's train_baseline(), already used there for
feature-importance reporting) -- that's the realistic production shape:
a model trained ahead of time and held in memory, not refit per request.
Held-out accuracy claims for that same architecture still come only from
defend/evaluation.py's cross-validated numbers; this module reuses the
model object, never the CV accuracy claim.
"""

import random
import time

import pandas as pd

from defend.constraint_drift import extract_domain
from defend.content_layer import score_injection_likelihood
from defend.gnn import predict_all_merchants
from defend.lightgbm_baseline import build_feature_matrix, train_baseline
from defend.rules import apply_rules
from generate.generated_sessions import load_cached_dataset
from mutator.mutate import JOINED_PATH, _load_mutator_cache

_HIGH_THRESHOLD = 0.5    # matches defend/llm_verdict.py's consistency_check thresholds
_MEDIUM_THRESHOLD = 0.15

SCHEMA_VERSION = "1.0"

SESSION_RISK_PAYLOAD_SCHEMA = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "string"},
        "agent_id": {"type": "string"},
        "intent_artifact_hash": {
            "type": "string",
            "description": "Ties this score to the specific signed Intent Artifact it rides along with.",
        },
        "session_risk_score": {
            "type": "number", "minimum": 0.0, "maximum": 1.0,
            "description": "max(lightgbm_prob, gnn_prob) -- the higher of the two attack-type-specific "
                            "supervised scores. A single number for a Decision Intelligence-style consumer "
                            "to threshold on; contributing_signals below is for explainability, not required "
                            "for the consuming system to act.",
        },
        "risk_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
        "contributing_signals": {
            "type": "object",
            "properties": {
                "rules_flagged": {"type": "boolean", "description": "Hard mandate-scope violation (Defend layer 1)."},
                "rules_reasons": {"type": "array", "items": {"type": "string"}},
                "lightgbm_prob": {"type": "number", "description": "Defend layer 2 -- prompt-injection classifier."},
                "content_score": {"type": "number", "description": "Defend layer 3 -- injection-phrasing similarity."},
                "gnn_prob": {
                    "type": ["number", "null"],
                    "description": "Defend layer 4 -- merchant-laundering-ring probability. "
                                    "null if the destination domain isn't a known merchant-network node.",
                },
            },
            "required": ["rules_flagged", "rules_reasons", "lightgbm_prob", "content_score", "gnn_prob"],
        },
    },
    "required": ["schema_version", "agent_id", "intent_artifact_hash", "session_risk_score",
                 "risk_level", "contributing_signals"],
}


def _all_sessions_by_id() -> dict:
    combined = {d["session"].agent_id: d["session"] for d in load_cached_dataset()}
    combined.update(_load_mutator_cache())
    return combined


def _fit_production_model():
    """One LightGBM fit on all 226 rows, held in memory -- the shape a
    real deployment would use (train offline, serve from memory), not
    refit per request. Returns (model, feature_matrix, agent_id_to_row)."""
    df = pd.read_csv(JOINED_PATH)
    X, y, categorical_cols = build_feature_matrix(df)
    model = train_baseline(X, y, categorical_cols)
    agent_id_to_row = {aid: i for i, aid in enumerate(df["agent_id"])}
    return model, X, agent_id_to_row


def _classify_risk_level(rules_flagged: bool, score: float, content_score: float) -> str:
    if rules_flagged or score >= _HIGH_THRESHOLD:
        return "HIGH"
    if score >= _MEDIUM_THRESHOLD or content_score >= _MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def build_session_risk_payload(agent_id: str, session, model, feature_row: pd.DataFrame, gnn_lookup: dict) -> dict:
    rules_flagged, rules_reasons = apply_rules(session)
    lightgbm_prob = float(model.predict(feature_row)[0])
    content_text = session.injection_payload_text or session.raw_utterance
    content_score = score_injection_likelihood(content_text)
    domain = extract_domain(session.task_origin_url)
    gnn_prob = gnn_lookup.get(domain)

    score = max(lightgbm_prob, gnn_prob or 0.0)
    risk_level = _classify_risk_level(rules_flagged, score, content_score)

    return {
        "schema_version": SCHEMA_VERSION,
        "agent_id": agent_id,
        "intent_artifact_hash": session.intent_artifact_hash,
        "session_risk_score": round(score, 4),
        "risk_level": risk_level,
        "contributing_signals": {
            "rules_flagged": rules_flagged,
            "rules_reasons": rules_reasons,
            "lightgbm_prob": round(lightgbm_prob, 4),
            "content_score": round(content_score, 4),
            "gnn_prob": round(gnn_prob, 4) if gnn_prob is not None else None,
        },
    }


def measure_latency(n_runs: int = 50, seed: int = 0) -> dict:
    """Real measured single-session, all-4-layers latency in milliseconds
    -- not an estimate. Model and GNN lookup are built once beforehand
    (the realistic in-memory-serving shape); only the per-session payload
    construction itself (rules + one LightGBM predict + content scoring +
    one dict lookup) is timed, one session at a time."""
    sessions = _all_sessions_by_id()
    model, X, agent_id_to_row = _fit_production_model()
    gnn_lookup = predict_all_merchants()

    agent_ids = [aid for aid in agent_id_to_row if aid in sessions]
    rng = random.Random(seed)
    sample_ids = rng.sample(agent_ids, min(n_runs, len(agent_ids)))

    # Warm-up call: LightGBM/pandas pay one-time cache/JIT costs on the
    # very first predict() that shouldn't be counted as steady-state latency.
    warmup_id = sample_ids[0]
    build_session_risk_payload(
        warmup_id, sessions[warmup_id], model, X.iloc[[agent_id_to_row[warmup_id]]], gnn_lookup
    )

    durations_ms = []
    for aid in sample_ids:
        row = X.iloc[[agent_id_to_row[aid]]]
        start = time.perf_counter()
        build_session_risk_payload(aid, sessions[aid], model, row, gnn_lookup)
        durations_ms.append((time.perf_counter() - start) * 1000)

    durations_ms.sort()
    n = len(durations_ms)
    return {
        "n_runs": n,
        "mean_ms": round(sum(durations_ms) / n, 3),
        "p50_ms": round(durations_ms[n // 2], 3),
        "p95_ms": round(durations_ms[min(int(n * 0.95), n - 1)], 3),
        "max_ms": round(durations_ms[-1], 3),
    }


if __name__ == "__main__":
    import json

    print("Measuring real end-to-end single-session latency (all 4 Defend layers)...")
    latency = measure_latency()
    print(json.dumps(latency, indent=2))

    sessions = _all_sessions_by_id()
    model, X, agent_id_to_row = _fit_production_model()
    gnn_lookup = predict_all_merchants()
    sample_id = next(aid for aid in agent_id_to_row if aid in sessions)
    example_payload = build_session_risk_payload(
        sample_id, sessions[sample_id], model, X.iloc[[agent_id_to_row[sample_id]]], gnn_lookup
    )
    print("\nExample payload:")
    print(json.dumps(example_payload, indent=2))
