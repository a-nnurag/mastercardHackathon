"""
LLM explanation layer — Defend layer 5 per TEAM_BRIEF.md Sec 4.6.

Architectural Invariants:
- The security decision is already finalized by RiskEngine before the LLM is invoked.
- The LLM cannot create, modify, lower, or override the decision or risk level.
- The LLM generates a plain-English explanation (RiskExplanation) only.
"""

import json
import os
from typing import Optional

import pandas as pd

from defend.constraint_drift import extract_domain
from defend.content_layer import score_injection_likelihood
from defend.contracts import RiskDecision, RiskExplanation, SignalResult, SignalSet
from defend.evaluation import cross_validated_predictions
from defend.gnn import predict_all_merchants
from defend.lightgbm_baseline import build_feature_matrix
from defend.model_registry import ModelRegistry
from defend.risk_engine import RiskEngine
from defend.rules import apply_rules
from generate.generated_sessions import load_cached_dataset
from generate.llm_adapter import GroqQuotaExhausted, OllamaAdapter, get_default_adapter

_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "joined_sessions.csv")

_EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "evidence_summary": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "evidence_summary"],
}

_EXPLANATION_PROMPT = """You are summarizing a fraud-risk explanation for a human risk analyst reviewing \
one AI shopping-agent session.

CRITICAL INVARIANTS:
1. The security decision is ALREADY FINALIZED by the deterministic/ML risk engine:
   - Decision: {decision}
   - Risk Level: {risk_level}
   - Risk Score: {risk_score:.3f}
2. Do NOT change, downgrade, or upgrade the decision.
3. Do NOT invent amounts, merchants, or evidence not stated below.
4. Explain ONLY the supplied decision, session facts, and evidence.

SESSION:
  What the human asked for: {raw_utterance}
  What actually got signed: {signed_artifact_text}
  Destination domain: {domain}

DETECTED EVIDENCE:
  - {evidence}

SIGNALS:
{signals}

Write a short plain-English explanation (2-3 sentences) summarizing why the system reached this \
{decision} ({risk_level}) decision, and list 1-3 concise evidence bullet points."""


def _get_adapter_with_fallback():
    try:
        adapter = get_default_adapter()
        adapter.generate_json("Reply with {\"ok\": true}", {"type": "object", "properties": {"ok": {"type": "boolean"}}})
        return adapter
    except (GroqQuotaExhausted, Exception):
        return OllamaAdapter()


def synthesize_explanation(
    decision: RiskDecision,
    session_summary: dict,
    adapter=None,
) -> RiskExplanation:
    """Generates a non-authoritative plain-English explanation for a finalized RiskDecision."""
    adapter = adapter or _get_adapter_with_fallback()
    prompt = _EXPLANATION_PROMPT.format(
        decision=decision.decision,
        risk_level=decision.risk_level,
        risk_score=decision.risk_score,
        raw_utterance=session_summary["raw_utterance"],
        signed_artifact_text=session_summary["signed_artifact_text"],
        domain=session_summary["domain"],
        evidence="\n  - ".join(decision.evidence) if decision.evidence else "None",
        signals=json.dumps(decision.signals, indent=2),
    )
    result = adapter.generate_json(prompt, _EXPLANATION_SCHEMA)
    return RiskExplanation(
        summary=result.get("summary", ""),
        evidence_summary=result.get("evidence_summary", []),
    )


def synthesize_verdict(
    session_summary: dict,
    rules_result: tuple,
    lightgbm_prob: float,
    content_score: float,
    gnn_prob: float | None,
    adapter=None,
) -> dict:
    """Compatibility wrapper that runs RiskEngine and returns structured verdict dictionary."""
    rules_flagged, rules_reasons = rules_result
    signal_set = SignalSet()
    signal_set.add(SignalResult(name="rules", value=rules_flagged, available=True, evidence=rules_reasons, hard_violation=rules_flagged))
    signal_set.add(SignalResult(name="lightgbm_prob", value=lightgbm_prob, available=True))
    signal_set.add(SignalResult(name="content_injection", value=content_score, available=True))
    signal_set.add(SignalResult(name="gnn_prob", value=gnn_prob, available=(gnn_prob is not None)))

    engine = RiskEngine()
    decision = engine.evaluate(signal_set)
    explanation = synthesize_explanation(decision, session_summary, adapter=adapter)

    rec_map = {"ALLOW": "ALLOW", "HOLD": "HOLD_FOR_REVIEW", "BLOCK": "BLOCK"}
    return {
        "risk_level": decision.risk_level,
        "recommendation": rec_map.get(decision.decision, decision.decision),
        "explanation": explanation.summary,
        "evidence_summary": explanation.evidence_summary,
        "decision": decision.decision,
        "risk_score": decision.risk_score,
    }


def consistency_check(
    verdict: dict,
    rules_flagged: bool,
    lightgbm_prob: float,
    content_score: float,
    gnn_prob: float | None,
) -> bool:
    """Sanity check: ensures risk_level consistency with numeric evidence."""
    signals_high = sum([
        rules_flagged,
        lightgbm_prob >= 0.5,
        content_score >= 0.15,
        (gnn_prob or 0) >= 0.5,
    ])
    if signals_high == 0:
        return verdict["risk_level"] == "LOW"
    if signals_high >= 2:
        return verdict["risk_level"] in ("MEDIUM", "HIGH")
    return True


def _lightgbm_prob_lookup() -> dict:
    if os.path.exists(_DATA_PATH):
        df = pd.read_csv(_DATA_PATH)
        X, y, categorical_cols = build_feature_matrix(df)
        oof = cross_validated_predictions(X, y, df["subtlety"], categorical_cols)
        return dict(zip(df["agent_id"], oof))
    return {}


def verdict_for_session(
    agent_id: str,
    adapter=None,
    lgb_lookup: dict = None,
    gnn_lookup: dict = None,
) -> dict:
    from mutator.mutate import _load_mutator_cache

    dataset = {d["session"].agent_id: d["session"] for d in load_cached_dataset()} if load_cached_dataset() else {}
    if not dataset:
        from generate.synthetic_sessions import all_sessions
        dataset = {s.agent_id: s for s in all_sessions()}
    try:
        dataset.update(_load_mutator_cache())
    except Exception:
        pass

    session = dataset[agent_id]

    if lgb_lookup is None:
        lgb_lookup = _lightgbm_prob_lookup()
    if gnn_lookup is None:
        gnn_model = ModelRegistry.get_gnn()
        if gnn_model is not None:
            gnn_lookup = predict_all_merchants(gnn_model)
        else:
            gnn_lookup = {}

    rules_result = apply_rules(session)
    lightgbm_prob = lgb_lookup.get(agent_id, 0.0)
    content_text = session.injection_payload_text or session.raw_utterance
    content_score = score_injection_likelihood(content_text)
    domain = extract_domain(session.task_origin_url)
    gnn_prob = gnn_lookup.get(domain)

    session_summary = {
        "raw_utterance": session.raw_utterance,
        "signed_artifact_text": session.signed_artifact_text,
        "domain": domain,
    }
    verdict = synthesize_verdict(session_summary, rules_result, lightgbm_prob, content_score, gnn_prob, adapter)
    return {
        "verdict": verdict,
        "signals": {
            "rules_flagged": rules_result[0], "rules_reasons": rules_result[1],
            "lightgbm_prob": lightgbm_prob, "content_score": content_score, "gnn_prob": gnn_prob,
        },
    }


if __name__ == "__main__":
    import random

    dataset = load_cached_dataset()
    if not dataset:
        from generate.synthetic_sessions import all_sessions
        sessions_list = all_sessions()
        dataset = [{"session": s, "subtlety": "obvious" if s.injection_present else "benign"} for s in sessions_list]

    sample = [d["session"].agent_id for d in dataset[:6]]
    lgb_lookup = _lightgbm_prob_lookup()
    gnn_model = ModelRegistry.get_gnn()
    gnn_lookup = predict_all_merchants(gnn_model) if gnn_model else {}
    adapter = _get_adapter_with_fallback()

    passed = 0
    for agent_id in sample:
        result = verdict_for_session(agent_id, adapter=adapter, lgb_lookup=lgb_lookup, gnn_lookup=gnn_lookup)
        v, s = result["verdict"], result["signals"]
        ok = consistency_check(v, s["rules_flagged"], s["lightgbm_prob"], s["content_score"], s["gnn_prob"])
        passed += ok
        print(f"\n{'=' * 70}\n{agent_id}  [{'OK' if ok else 'INCONSISTENT'}]")
        print(f"  signals: rules={s['rules_flagged']} lgb={s['lightgbm_prob']:.3f} "
              f"content={s['content_score']:.3f} gnn={s['gnn_prob']}")
        print(f"  risk_level={v['risk_level']}  recommendation={v['recommendation']}")
        print(f"  explanation: {v['explanation']}")

    print(f"\n{'=' * 70}\nConsistency: {passed}/{len(sample)}")
