import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate.narrative_generator import _base_session, _mandate

_FIELDS = {
    "raw_utterance": "Book me a flight to Bangalore under 8000 rupees.",
    "signed_artifact_text": "Book economy flight to Bangalore, budget cap INR 8000.",
    "task_origin_url": "https://cleartrip.com/flights",
    "content_sources_ingested": ["cleartrip.com"],
}


def test_hops_since_intent_not_conditioned_on_injection_present():
    # Regression test for a real label-leakage bug: an earlier version set
    # hops_since_intent=0 for every benign session and random(1,4) for
    # every hijacked one, letting a LightGBM model "detect" hijacks with
    # perfect AUC just by reading this one bookkeeping field. Both branches
    # must draw from the same range now — sample enough times that a
    # reintroduced 0-only-when-benign / never-0-when-hijacked pattern
    # would almost certainly get caught.
    mandate = _mandate("travel")

    benign_values = {_base_session("travel", mandate, _FIELDS, injection_present=False).hops_since_intent for _ in range(50)}
    hijacked_values = {_base_session("travel", mandate, _FIELDS, injection_present=True).hops_since_intent for _ in range(50)}

    assert 0 in hijacked_values, "hops_since_intent should be able to be 0 even when injection_present=True"
    assert benign_values & hijacked_values, "benign and hijacked sessions should draw from overlapping ranges"
