"""
Day 2 kill-criterion test.

Question: does utterance_artifact_divergence actually separate hijacked
sessions from legitimate ones? If not — pivot immediately to the fallback
signal pair (constraint_drift + ingestion_source_trust_score), per the
plan's kill criteria. Don't force a signal that doesn't separate.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import roc_auc_score

from generate.synthetic_sessions import all_sessions
from defend.divergence import score_sessions


def _report(name, sessions, labels, get_score):
    scores = [get_score(s) for s in sessions]
    if any(s is None for s in scores):
        print(f"\n{name}: unavailable (model not loaded — see divergence.py)")
        return None
    legit = [sc for sc, s in zip(scores, sessions) if not s.injection_present]
    hijack = [sc for sc, s in zip(scores, sessions) if s.injection_present]
    auc = roc_auc_score(labels, scores)
    print(f"\n{name}:")
    print(f"  legit  mean={np.mean(legit):.3f} std={np.std(legit):.3f}")
    print(f"  hijack mean={np.mean(hijack):.3f} std={np.std(hijack):.3f}")
    print(f"  AUC = {auc:.3f}")
    return auc


def run():
    sessions = score_sessions(all_sessions())
    labels = [int(s.injection_present) for s in sessions]

    print("=" * 60)
    print("DAY 2 ISOLATION TEST — utterance_artifact_divergence")
    print("=" * 60)

    lexical_auc = _report("LEXICAL (TF-IDF)", sessions, labels, lambda s: s.lexical_divergence)
    semantic_auc = _report("SEMANTIC (embeddings)", sessions, labels, lambda s: s.semantic_divergence)
    auc = _report("COMBINED (what the pipeline uses)", sessions, labels,
                   lambda s: s.utterance_artifact_divergence)

    print("\n" + "-" * 60)
    if semantic_auc is None:
        print("Semantic model didn't load — this run only proves the lexical")
        print("signal, which is known to be weak on subtle attacks (see")
        print("divergence.py docstring). Rerun in an environment with model")
        print("access before treating any result here as the real gate.")
        print("-" * 60)
        return auc
    if auc >= 0.85:
        print("READ: strong separation on this synthetic set.")
        print("BUT — this is TF-IDF on hand-written examples (see divergence.py")
        print("docstring). This validates the MECHANISM, not the real signal.")
        print("Do not treat this as the real Day 2 gate result.")
    elif auc >= 0.65:
        print("READ: partial separation — signal exists but is noisy.")
        print("Even on easy hand-written examples this is marginal. Expect a")
        print("real embedding model + real narrative-generator attacks to be")
        print("harder to separate, not easier. Investigate before committing.")
    else:
        print("READ: no meaningful separation, even on easy examples.")
        print("Per kill criteria: pivot to constraint_drift + ")
        print("ingestion_source_trust_score as the primary signal now.")
    print("-" * 60)

    return auc


if __name__ == "__main__":
    run()
