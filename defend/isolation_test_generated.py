"""
Task 2's actual definition of done: rerun the isolation test against the
real LLM-generated session set (generate/generated_sessions.py), split
into obvious vs subtle hijack subsets, AUC reported separately for each —
not just overall. The subtle subset is the one that actually matters
(same-domain, only amount/merchant/destination shifted); a low subtle AUC
here is the real finding this test exists to surface, not a bug to hide.

Also reports the Task 3 fallback signals (constraint_drift,
ingestion_source_trust_score) on the same sessions, since the honest
comparison between "headline signal" and "fallback signal" is the actual
point of building both — not just the divergence number in isolation.

Loads whatever's in the cache via load_cached_dataset() — deliberately
does NOT call generate_dataset(), so running this evaluation never has
the side effect of spending more LLM API calls. Run
`generate/generated_sessions.py` directly first to (re)generate.

Kept separate from isolation_test.py (which plan.md treats as the
already-correct Task 1 gate on the hand-written set) rather than folding
this in — different input source, same underlying scoring
(defend/divergence.py's score_sessions(), unchanged).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import roc_auc_score

from generate.generated_sessions import load_cached_dataset
from defend.divergence import score_sessions
from defend.constraint_drift import compute_constraint_drift, compute_ingestion_source_trust_score


def _report(name, sessions, get_score):
    labels = [int(s.injection_present) for s in sessions]
    scores = [get_score(s) for s in sessions]
    if any(s is None for s in scores) or len(set(labels)) < 2:
        print(f"\n{name}: unavailable (no model, or only one class present)")
        return None
    legit = [sc for sc, lbl in zip(scores, labels) if lbl == 0]
    hijack = [sc for sc, lbl in zip(scores, labels) if lbl == 1]
    auc = roc_auc_score(labels, scores)
    print(f"\n{name} (n={len(sessions)}):")
    print(f"  legit  mean={np.mean(legit):.3f} std={np.std(legit):.3f} (n={len(legit)})")
    print(f"  hijack mean={np.mean(hijack):.3f} std={np.std(hijack):.3f} (n={len(hijack)})")
    print(f"  AUC = {auc:.3f}")
    return auc


def _run_signal(signal_name, get_score, sessions, subtlety):
    legit_sessions = [s for s, t in zip(sessions, subtlety) if t == "n/a"]
    obvious_sessions = [s for s, t in zip(sessions, subtlety) if t == "obvious"] + legit_sessions
    subtle_sessions = [s for s, t in zip(sessions, subtlety) if t == "subtle"] + legit_sessions

    print("=" * 60)
    print(f"SIGNAL: {signal_name}")
    print("=" * 60)
    overall_auc = _report("OVERALL (all hijacked vs all legit)", sessions, get_score)
    obvious_auc = _report("OBVIOUS subset vs legit", obvious_sessions, get_score)
    subtle_auc = _report("SUBTLE subset vs legit", subtle_sessions, get_score)

    print("\n" + "-" * 60)
    print(f"Overall AUC={overall_auc:.3f}  Obvious AUC={obvious_auc:.3f}  Subtle AUC={subtle_auc:.3f}")
    if subtle_auc is not None and obvious_auc is not None and subtle_auc < obvious_auc - 0.1:
        print("READ: subtle attacks separate meaningfully worse than obvious ones.")
        print("This is a real finding — report it as-is, don't average it away.")
    print("-" * 60)

    return {"overall": overall_auc, "obvious": obvious_auc, "subtle": subtle_auc}


def run():
    dataset = load_cached_dataset()
    if not dataset:
        print("No cached sessions found. Run `python3 -m generate.generated_sessions` first.")
        return {}

    sessions = score_sessions([d["session"] for d in dataset])
    for s in sessions:
        s.constraint_drift = compute_constraint_drift(s)
        s.ingestion_source_trust_score = compute_ingestion_source_trust_score(s)
    subtlety = [d["subtlety"] for d in dataset]

    print(f"Evaluating {len(sessions)} cached sessions "
          f"({sum(1 for t in subtlety if t == 'n/a')} legit, "
          f"{sum(1 for t in subtlety if t == 'obvious')} obvious-hijack, "
          f"{sum(1 for t in subtlety if t == 'subtle')} subtle-hijack)\n")

    results = {
        "utterance_artifact_divergence": _run_signal(
            "utterance_artifact_divergence", lambda s: s.utterance_artifact_divergence, sessions, subtlety),
        "constraint_drift": _run_signal(
            "constraint_drift", lambda s: s.constraint_drift, sessions, subtlety),
        "ingestion_source_trust_score": _run_signal(
            "ingestion_source_trust_score", lambda s: s.ingestion_source_trust_score, sessions, subtlety),
    }
    return results


if __name__ == "__main__":
    run()
