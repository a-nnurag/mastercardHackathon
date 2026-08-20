import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from dashboard.data_access import (
    build_session_index,
    compute_attack1_metrics,
    load_all_sessions,
    load_mutation_rounds,
)


def test_load_all_sessions_combines_base_and_mutator_with_no_duplicates():
    sessions, subtlety_list = load_all_sessions()
    agent_ids = [s.agent_id for s in sessions]

    assert len(agent_ids) == len(set(agent_ids)), "Mutator sessions must override/dedupe by agent_id, not duplicate"
    assert len(sessions) == len(subtlety_list)
    assert len(sessions) >= 210  # at least the base Task 2 dataset


@pytest.fixture(scope="module")
def session_index():
    # Real pipeline calls (LightGBM CV + GNN training), run once and
    # shared across tests in this file -- mirrors tests/test_gnn.py's
    # own "run the real thing, assert on shape/ranges" style rather than
    # mocking signals that are cheap to actually compute.
    return build_session_index()


def test_build_session_index_row_shape_and_signal_ranges(session_index):
    rows, lgb_lookup, gnn_lookup = session_index
    assert len(rows) >= 210
    assert isinstance(lgb_lookup, dict)
    assert isinstance(gnn_lookup, dict)

    for row in rows:
        assert row["agent_id"]
        assert row["subtlety"] in ("benign", "obvious", "subtle")
        signals = row["signals"]
        assert isinstance(signals["rules_flagged"], bool)
        assert 0.0 <= signals["content_score"] <= 1.0
        if signals["lightgbm_prob"] is not None:
            assert 0.0 <= signals["lightgbm_prob"] <= 1.0
        if signals["gnn_prob"] is not None:
            assert 0.0 <= signals["gnn_prob"] <= 1.0


def test_build_session_index_agent_ids_are_unique(session_index):
    rows, _, _ = session_index
    agent_ids = [r["agent_id"] for r in rows]
    assert len(agent_ids) == len(set(agent_ids))


def test_compute_attack1_metrics_returns_real_slices():
    metrics = compute_attack1_metrics()
    assert metrics["overall"] is not None
    for key in ("precision", "recall", "f1", "auc"):
        assert 0.0 <= metrics["overall"][key] <= 1.0
    assert metrics["fp_rate_benign"] is None or 0.0 <= metrics["fp_rate_benign"] <= 1.0


def test_load_mutation_rounds_matches_documented_task10_results():
    rounds = load_mutation_rounds()
    if not rounds:
        pytest.skip("data/mutation_rounds.json not present in this clone")
    assert [r["round"] for r in rounds] == list(range(1, len(rounds) + 1))
    for r in rounds:
        assert 0.0 <= r["precision"] <= 1.0
        assert 0.0 <= r["recall"] <= 1.0
