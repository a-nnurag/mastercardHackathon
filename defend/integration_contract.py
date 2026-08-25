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
from typing import Any, Optional

import pandas as pd

from defend.constraint_drift import ConstraintDriftDetector, extract_domain
from defend.content_layer import ContentInjectionDetector, score_injection_likelihood
from defend.contracts import SignalResult, SignalSet
from defend.detection_context import DetectionContext
from defend.gnn import GNNDetector, predict_all_merchants
from defend.lightgbm_baseline import LightGBMDetector, build_feature_matrix, train_baseline
from defend.model_registry import ModelRegistry
from defend.risk_engine import RiskEngine
from defend.rules import RuleDetector, apply_rules
from generate.generated_sessions import load_cached_dataset
from generate.session_schema import AgentSession
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
            "description": "Supervised fusion score produced by RiskEngine.",
        },
        "risk_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
        "decision": {"type": "string", "enum": ["ALLOW", "HOLD", "BLOCK"]},
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
    dataset = load_cached_dataset()
    combined = {d["session"].agent_id: d["session"] for d in dataset} if dataset else {}
    if not combined:
        from generate.synthetic_sessions import all_sessions
        combined = {s.agent_id: s for s in all_sessions()}
    try:
        combined.update(_load_mutator_cache())
    except Exception:
        pass
    return combined


def _get_production_models_and_features():
    """Loads or fits production models for serving. Returns (lgb_model, gnn_model, X, agent_id_to_row, gnn_lookup)."""
    import os
    if os.path.exists(JOINED_PATH):
        df = pd.read_csv(JOINED_PATH)
    else:
        from defend.training.lightgbm import _create_synthetic_joined_dataset
        df = _create_synthetic_joined_dataset()

    X, y, categorical_cols = build_feature_matrix(df)
    agent_id_to_row = {aid: i for i, aid in enumerate(df["agent_id"])}

    lgb_model = ModelRegistry.get_lightgbm()
    if lgb_model is None:
        lgb_model = train_baseline(X, y, categorical_cols)

    gnn_model = ModelRegistry.get_gnn()
    if gnn_model is None:
        from defend.gnn import MerchantGNN
        gnn_model = MerchantGNN(in_dim=4)
        gnn_model.eval()

    gnn_lookup = predict_all_merchants(gnn_model)
    return lgb_model, gnn_model, X, agent_id_to_row, gnn_lookup


from defend.pipeline import DetectionPipeline


def build_session_risk_payload(
    agent_id: str,
    session: AgentSession,
    model: Optional[Any] = None,
    feature_row: Optional[pd.DataFrame] = None,
    gnn_lookup: Optional[dict] = None,
    engine: Optional[RiskEngine] = None,
    pipeline: Optional[DetectionPipeline] = None,
) -> dict:
    """Builds standard session-risk payload using DetectionPipeline and RiskEngine."""
    engine = engine or RiskEngine()
    context = DetectionContext(
        tabular_model=model,
        feature_row=feature_row,
        gnn_merchant_lookup=gnn_lookup or {},
    )

    pipe = pipeline or DetectionPipeline(context=context)
    signal_set = pipe.run(session, context)
    decision = engine.evaluate(signal_set)

    rules_sig = signal_set.get("rules")
    rules_flagged = bool(rules_sig.value) if rules_sig else False
    rules_reasons = rules_sig.evidence if rules_sig else []

    lgb_val = signal_set.get_value("lightgbm_prob", 0.0)
    content_val = signal_set.get_value("content_injection", 0.0)
    gnn_val = signal_set.get_value("gnn_prob", None)

    return {
        "schema_version": SCHEMA_VERSION,
        "agent_id": agent_id,
        "intent_artifact_hash": session.intent_artifact_hash,
        "session_risk_score": round(decision.risk_score, 4),
        "risk_level": decision.risk_level,
        "decision": decision.decision,
        "contributing_signals": {
            "rules_flagged": rules_flagged,
            "rules_reasons": rules_reasons,
            "lightgbm_prob": round(float(lgb_val), 4) if lgb_val is not None else 0.0,
            "content_score": round(float(content_val), 4) if content_val is not None else 0.0,
            "gnn_prob": round(float(gnn_val), 4) if gnn_val is not None else None,
        },
        "evidence": decision.evidence,
        "attack_family": decision.attack_family,
    }


def measure_latency(n_runs: int = 50, seed: int = 0) -> dict:
    """Real measured single-session latency in milliseconds (no training in loop)."""
    sessions = _all_sessions_by_id()
    lgb_model, gnn_model, X, agent_id_to_row, gnn_lookup = _get_production_models_and_features()
    engine = RiskEngine()

    agent_ids = [aid for aid in agent_id_to_row if aid in sessions]
    rng = random.Random(seed)
    sample_ids = rng.sample(agent_ids, min(n_runs, len(agent_ids)))

    # Warm-up call
    warmup_id = sample_ids[0]
    build_session_risk_payload(
        warmup_id, sessions[warmup_id], lgb_model, X.iloc[[agent_id_to_row[warmup_id]]], gnn_lookup, engine=engine
    )

    durations_ms = []
    for aid in sample_ids:
        row = X.iloc[[agent_id_to_row[aid]]]
        start = time.perf_counter()
        build_session_risk_payload(aid, sessions[aid], lgb_model, row, gnn_lookup, engine=engine)
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

    print("Measuring real end-to-end single-session latency (all Defend layers + RiskEngine)...")
    latency = measure_latency()
    print(json.dumps(latency, indent=2))

    sessions = _all_sessions_by_id()
    lgb_model, gnn_model, X, agent_id_to_row, gnn_lookup = _get_production_models_and_features()
    sample_id = next(aid for aid in agent_id_to_row if aid in sessions)
    example_payload = build_session_risk_payload(
        sample_id, sessions[sample_id], lgb_model, X.iloc[[agent_id_to_row[sample_id]]], gnn_lookup
    )
    print("\nExample payload:")
    print(json.dumps(example_payload, indent=2))
