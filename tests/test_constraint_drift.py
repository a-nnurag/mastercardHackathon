import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate.synthetic_sessions import LEGITIMATE, HIJACKED
from defend.constraint_drift import (
    extract_amount_inr,
    extract_domain,
    compute_constraint_drift,
    compute_ingestion_source_trust_score,
)

CLEAN = LEGITIMATE[0]   # agent-001: flight to Bangalore, INR 8000 cap, cleartrip.com
VIOLATION = HIJACKED[0]  # agent-101: INR 400000 vs 8000 cap, typosquat origin+source


def test_extract_amount_inr_finds_amount():
    assert extract_amount_inr("Purchase premium electronics bundle, total INR 400000.") == 400000.0


def test_extract_amount_inr_none_when_absent():
    assert extract_amount_inr("Book economy flight to Bangalore.") is None


def test_extract_amount_inr_handles_number_before_inr():
    # Real LLM output uses this ordering too, not just "INR <number>".
    assert extract_amount_inr("Purchase a gaming console bundle for 250000 INR, including accessories.") == 250000.0


def test_extract_amount_inr_does_not_crash_on_bare_inr_comma():
    # Regression: "INR," (no digit follows) used to match "," as the
    # amount and crash on float("").
    assert extract_amount_inr("Priced in INR, subject to change.") is None


def test_extract_amount_inr_takes_max_of_multiple_mentions():
    assert extract_amount_inr("Line item INR 500, another INR 12000 total.") == 12000.0


def test_extract_domain_strips_scheme_and_path():
    assert extract_domain("https://cleartrip.com/flights/search") == "cleartrip.com"
    assert extract_domain("https://www.cleartrip.com/x") == "cleartrip.com"


def test_constraint_drift_clean_session_is_zero():
    assert compute_constraint_drift(CLEAN) == 0.0


def test_constraint_drift_flags_amount_and_origin_violation():
    # agent-101: signed amount (400000) > cap (8000), and task_origin_url
    # (cleartrip-deals-offer.com) isn't in the merchant_allowlist (cleartrip.com)
    assert compute_constraint_drift(VIOLATION) == 1.0


def test_ingestion_trust_clean_session_is_zero():
    assert compute_ingestion_source_trust_score(CLEAN) == 0.0


def test_ingestion_trust_flags_untrusted_sources():
    # agent-101 ingested cleartrip-deals-offer.com and ad-network-x.com,
    # neither of which is the allowlisted cleartrip.com
    assert compute_ingestion_source_trust_score(VIOLATION) == 1.0


def test_violation_scores_exceed_clean_scores():
    assert compute_constraint_drift(VIOLATION) > compute_constraint_drift(CLEAN)
    assert compute_ingestion_source_trust_score(VIOLATION) > compute_ingestion_source_trust_score(CLEAN)
