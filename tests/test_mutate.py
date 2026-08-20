import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from identify.schema import AttackVector


def test_attack_vector_reviewed_defaults_true_for_old_entries():
    # The 61 entries discovered before Task 10 added this field don't
    # carry a "reviewed" key at all -- from_dict must default them to
    # True rather than crashing or defaulting to False (which would
    # wrongly mark every pre-existing entry as an unvetted candidate).
    old_entry = {
        "name": "Branded Whisper Attack", "mechanism": "...", "defend_signal": "...",
        "category": "merchant impersonation", "delivery_vector": "typosquat-url",
        "mechanism_type": "pre-signature-corruption", "grounding": ["https://example.com"],
    }
    restored = AttackVector.from_dict(old_entry)
    assert restored.reviewed is True


def test_attack_vector_reviewed_false_for_mutator_candidates():
    new_entry = {
        "name": "Mutator round 2 variant", "mechanism": "...", "defend_signal": "...",
        "category": "prompt injection", "delivery_vector": "closer-wording-subtle",
        "mechanism_type": "pre-signature-corruption", "grounding": ["mutator:round_2:agent-x"],
        "reviewed": False,
    }
    restored = AttackVector.from_dict(new_entry)
    assert restored.reviewed is False
    assert restored.to_dict()["reviewed"] is False


def test_find_hard_sessions_identifies_low_confidence_and_false_negatives():
    from mutator.mutate import find_hard_sessions

    # Minimal fixture matching build_feature_matrix's required columns,
    # with a hand-picked feature (constraint_drift) that's perfectly
    # predictive, so we can assert on which specific rows come out "hard"
    # rather than fighting real LightGBM randomness.
    base_row = {
        "TransactionID": 1, "ieee_cis_isFraud": 0, "agent_registry_status": "valid",
        "mandate_categories": "['travel']", "mandate_merchant_allowlist": "['cleartrip.com']",
        "content_sources_ingested": "['cleartrip.com']", "task_origin_url": "https://cleartrip.com",
        "utterance_artifact_divergence": 0.1, "ingestion_source_trust_score": 0.0,
        "hops_since_intent": 0, "tool_calls_made": 3,
    }
    rows = []
    for i in range(20):
        row = dict(base_row)
        row["agent_id"] = f"agent-{i}"
        is_hijack = i % 2 == 0
        row["injection_present"] = is_hijack
        row["subtlety"] = "obvious" if is_hijack else "benign"
        # Rows 0 and 2 are hijacked but LOOK clean (constraint_drift=0) --
        # these should show up as hard; the rest of the hijacked rows are
        # obviously flagged (constraint_drift=1).
        row["constraint_drift"] = 0.0 if i in (0, 2) else (1.0 if is_hijack else 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    hard = find_hard_sessions(df, threshold=0.65)
    hard_ids = {agent_id for agent_id, _ in hard}

    assert "agent-0" in hard_ids
    assert "agent-2" in hard_ids
    assert "agent-4" not in hard_ids  # an obviously-flagged hijack, should score high
