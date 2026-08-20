"""
constraint_drift + ingestion_source_trust_score — fallback primary signal.

Per plan.md Task 3: built in parallel with the divergence signal, not only
as an emergency fallback. Unlike utterance_artifact_divergence (an ML/NLP
similarity score), these are deliberately simple rules checks against
fields already present on AgentSession/MandateScope — low-risk, easy to
explain to a risk analyst, no embedding model dependency.

AgentSession's schema is locked (session_schema.py) and has no separate
structured "actual amount" / "actual merchant" field — only free-text
signed_artifact_text and task_origin_url. So both functions here work by
extracting signal from that text/URL rather than assuming structured
fields that don't exist. That means the amount check only fires when an
INR amount is actually present in signed_artifact_text (returns no
violation, not a false one, when the text doesn't state an amount) — this
is a known limitation of working against free text, not a bug.
"""

import re
from difflib import SequenceMatcher
from urllib.parse import urlparse

from generate.session_schema import AgentSession

_AMOUNT_RE = re.compile(
    r"(?:INR\s*(\d[\d,]*(?:\.\d+)?))|(?:(\d[\d,]*(?:\.\d+)?)\s*INR)", re.IGNORECASE
)


def extract_amount_inr(text: str) -> float | None:
    """Pulls the largest INR amount out of free text, in either "INR 3000"
    or "3000 INR" order (real LLM output uses both). None if none found.
    Takes the max rather than the first match since a session can mention
    several figures (line items, a running total) and the largest is the
    one that matters for a cap check. Requires a leading digit so "INR,"
    (comma with no digit) doesn't parse as an amount."""
    amounts = [float((a or b).replace(",", "")) for a, b in _AMOUNT_RE.findall(text)]
    return max(amounts) if amounts else None


def extract_domain(url: str) -> str:
    """Registrable-ish domain from a URL, e.g. https://a.b.com/x -> a.b.com."""
    netloc = urlparse(url).netloc or urlparse(f"//{url}").netloc
    return netloc.lower().removeprefix("www.")


def compute_constraint_drift(session: AgentSession) -> float:
    """
    0 = no detected violation of mandate_scope, 1 = every checkable
    constraint was violated. Checks run:
      - amount in signed_artifact_text (if any) over mandate_scope.amount_cap
      - task_origin_url domain not in mandate_scope.merchant_allowlist
    Both checks always run (2 checks), so the score is violations/2.
    """
    violations = 0

    amount = extract_amount_inr(session.signed_artifact_text)
    if amount is not None and amount > session.mandate_scope.amount_cap:
        violations += 1

    origin_domain = extract_domain(session.task_origin_url)
    allowlist = {extract_domain(d) for d in session.mandate_scope.merchant_allowlist}
    if origin_domain not in allowlist:
        violations += 1

    return violations / 2


def compute_ingestion_source_trust_score(session: AgentSession) -> float:
    """
    0 = every ingested source is trusted, 1 = every ingested source looks
    untrustworthy. A source counts as trusted only on an exact domain match
    against mandate_scope.merchant_allowlist. A domain that's textually
    *close* to an allowlisted one (typosquat) or far from all of them
    (unknown) both count as untrusted — this is a domain-age-proxy
    heuristic (string edit-distance), not a real WHOIS/threat-intel lookup.
    """
    sources = session.content_sources_ingested
    if not sources:
        return 0.0

    allowlist = [extract_domain(d) for d in session.mandate_scope.merchant_allowlist]
    untrusted = 0
    for source in sources:
        domain = extract_domain(source)
        if domain in allowlist:
            continue
        untrusted += 1

    return untrusted / len(sources)


def typosquat_similarity(domain: str, allowlist: list[str]) -> float:
    """
    Max SequenceMatcher ratio of `domain` against any allowlisted domain.
    High similarity (close to but not exactly 1.0) without an exact match
    is the typosquat signature this module is meant to catch — exposed
    separately from the trust score above for inspection/debugging.
    """
    if not allowlist:
        return 0.0
    return max(SequenceMatcher(None, domain, d).ratio() for d in allowlist)
