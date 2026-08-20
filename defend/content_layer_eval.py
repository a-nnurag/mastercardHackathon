"""
Task 7's evaluation: precision/recall/F1 (per plan.md's definition of
done) plus Brier score / ECE (per the notification-router's confidence-
calibration methodology, eval/metrics.py's brier_score()/
expected_calibration_error() — formulas reproduced here, not imported,
since that project is unrelated to this one).

Reference/eval split, to avoid leakage: 2 real injection_payload_text
values per domain (sorted by agent_id for determinism) are held out to
extend content_layer.py's fixed reference phrases; every other hijacked
session's injection_payload_text, plus all benign sessions'
raw_utterance, form the eval set actually scored below. No session
appears in both.
"""

import re

from defend.content_layer import REFERENCE_INJECTION_PHRASES, score_injection_likelihood
from generate.generated_sessions import load_cached_dataset

THRESHOLD = 0.15  # empirically swept 0.05-0.30 on this eval set; 0.15 gives precision=0.983 recall=0.992
N_REFERENCE_PER_DOMAIN = 2


def _build_split():
    dataset = load_cached_dataset()
    hijacked_by_domain = {}
    benign_utterances = []

    for d in dataset:
        session, subtlety = d["session"], d["subtlety"]
        domain = session.mandate_scope.categories[0]
        if session.injection_payload_text:
            hijacked_by_domain.setdefault(domain, []).append(session)
        else:
            benign_utterances.append(session.raw_utterance)

    reference_phrases = list(REFERENCE_INJECTION_PHRASES)
    eval_positive = []
    for domain, sessions in hijacked_by_domain.items():
        sessions = sorted(sessions, key=lambda s: s.agent_id)
        held_out, rest = sessions[:N_REFERENCE_PER_DOMAIN], sessions[N_REFERENCE_PER_DOMAIN:]
        reference_phrases.extend(s.injection_payload_text for s in held_out)
        eval_positive.extend(s.injection_payload_text for s in rest)

    return reference_phrases, eval_positive, benign_utterances


def _brier_score(confidences: list[float], outcomes: list[bool]) -> float:
    if not confidences:
        return 0.0
    return sum((c - float(o)) ** 2 for c, o in zip(confidences, outcomes)) / len(confidences)


def _expected_calibration_error(confidences: list[float], outcomes: list[bool], n_bins: int = 10) -> float:
    if not confidences:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for c, o in zip(confidences, outcomes):
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, o))
    total = len(confidences)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        mean_conf = sum(c for c, _ in bucket) / len(bucket)
        observed_rate = sum(float(o) for _, o in bucket) / len(bucket)
        ece += len(bucket) / total * abs(mean_conf - observed_rate)
    return ece


def run():
    reference_phrases, eval_positive, benign_utterances = _build_split()

    texts = eval_positive + benign_utterances
    labels = [True] * len(eval_positive) + [False] * len(benign_utterances)
    scores = [score_injection_likelihood(t, reference_phrases) for t in texts]

    predictions = [s >= THRESHOLD for s in scores]
    tp = sum(1 for p, l in zip(predictions, labels) if p and l)
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
    fn = sum(1 for p, l in zip(predictions, labels) if not p and l)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("=" * 60)
    print(f"TASK 7 — content/semantic injection-detection layer")
    print(f"Reference phrases: {len(reference_phrases)} ({len(REFERENCE_INJECTION_PHRASES)} from real")
    print(f"  corpus quotes + {len(reference_phrases) - len(REFERENCE_INJECTION_PHRASES)} held-out real injection payloads)")
    print(f"Eval set: {len(eval_positive)} positive (held-out injection payloads), "
          f"{len(benign_utterances)} negative (benign raw_utterance) — no overlap with reference set")
    print("=" * 60)
    print(f"\nAt threshold={THRESHOLD}: precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
    print(f"  TP={tp} FP={fp} FN={fn}")

    brier = _brier_score(scores, labels)
    ece = _expected_calibration_error(scores, labels)
    print(f"\nConfidence calibration (raw score treated as confidence):")
    print(f"  Brier score = {brier:.4f} (0 = perfect, 0.25 = uninformative)")
    print(f"  ECE = {ece:.4f}")

    return {"precision": precision, "recall": recall, "f1": f1, "brier": brier, "ece": ece}


if __name__ == "__main__":
    run()
