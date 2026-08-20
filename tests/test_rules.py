import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate.synthetic_sessions import LEGITIMATE, HIJACKED
from defend.rules import apply_rules

CLEAN = LEGITIMATE[0]    # agent-001: flight to Bangalore, INR 8000 cap, cleartrip.com
VIOLATION = HIJACKED[0]  # agent-101: INR 400000 vs 8000 cap, typosquat origin


def test_clean_session_not_flagged():
    flagged, reasons = apply_rules(CLEAN)
    assert flagged is False
    assert reasons == []


def test_violation_session_flagged_with_reasons():
    flagged, reasons = apply_rules(VIOLATION)
    assert flagged is True
    assert len(reasons) == 2  # amount breach + off-allowlist origin
    assert any("amount cap breached" in r for r in reasons)
    assert any("not in merchant allowlist" in r for r in reasons)


def test_registry_status_check_fires_independently():
    import dataclasses

    revoked = dataclasses.replace(CLEAN, agent_registry_status="revoked")
    flagged, reasons = apply_rules(revoked)
    assert flagged is True
    assert any("agent_registry_status" in r for r in reasons)


def test_all_hand_written_hijacked_sessions_flagged():
    # Definition-of-done sanity check from plan.md Task 4, on the existing
    # hand-written set: rules alone should catch the blatant cases.
    for session in HIJACKED:
        flagged, _ = apply_rules(session)
        assert flagged is True, f"{session.agent_id} should have been flagged"


def test_all_hand_written_legitimate_sessions_clean():
    for session in LEGITIMATE:
        flagged, _ = apply_rules(session)
        assert flagged is False, f"{session.agent_id} should not have been flagged"
