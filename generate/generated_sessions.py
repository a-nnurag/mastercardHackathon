"""
Orchestrates narrative_generator.py across all domains and both hijack
subtlety levels, and caches the result to data/generated_sessions.json —
per plan.md: "don't regenerate on every test run, LLM calls cost money and
time."

Each cached entry is a full AgentSession (a superset of the locked
to_row(), which is deliberately narrower for the IEEE-CIS join) plus a
`subtlety` tag ("n/a" for benign sessions). Subtlety lives here, in the
cache, not as a field on AgentSession — it's an artifact of how a session
was generated, not part of the locked session schema.
"""

import dataclasses
import json
import os

from generate.narrative_generator import DOMAINS, generate_benign_session, generate_hijacked_session
from generate.session_schema import AgentSession, MandateScope

CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "generated_sessions.json")


def _session_to_dict(session: AgentSession, subtlety: str) -> dict:
    row = dataclasses.asdict(session)
    row["mandate_scope"]["expiry"] = session.mandate_scope.expiry.isoformat()
    row["subtlety"] = subtlety
    return row


def _session_from_dict(row: dict) -> tuple[AgentSession, str]:
    row = dict(row)
    subtlety = row.pop("subtlety")
    mandate_dict = dict(row.pop("mandate_scope"))
    from datetime import datetime
    mandate_dict["expiry"] = datetime.fromisoformat(mandate_dict["expiry"])
    mandate = MandateScope(**mandate_dict)
    session = AgentSession(mandate_scope=mandate, **row)
    return session, subtlety


def load_cached_dataset() -> list[dict]:
    """
    Loads whatever's currently in the cache, as-is — never triggers
    generation. For evaluation scripts, which shouldn't have the side
    effect of spending API calls just by being run. Returns [] if no
    cache file exists yet.
    """
    if not os.path.exists(CACHE_PATH):
        return []
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cached_rows = json.load(f)
    return [dict(zip(("session", "subtlety"), _session_from_dict(row))) for row in cached_rows]


def generate_dataset(n_per_slot: int = 7, force_regenerate: bool = False) -> list[dict]:
    """
    Returns a list of {"session": AgentSession, "subtlety": str} dicts,
    generating fresh via the LLM only if no cache exists (or
    force_regenerate=True). Slots: domain x {benign, hijacked-obvious,
    hijacked-subtle} x n_per_slot.
    """
    slots = [
        (domain, kind, subtlety)
        for domain in DOMAINS
        for kind, subtlety in (("benign", "n/a"), ("hijacked", "obvious"), ("hijacked", "subtle"))
        for _ in range(n_per_slot)
    ]

    results = []
    if os.path.exists(CACHE_PATH) and not force_regenerate:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cached_rows = json.load(f)
        results = [dict(zip(("session", "subtlety"), _session_from_dict(row))) for row in cached_rows]
        if len(results) >= len(slots):
            return results[: len(slots)]
        print(f"Resuming: {len(results)}/{len(slots)} sessions already cached, generating the rest.")

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    def _save():
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump([_session_to_dict(r["session"], r["subtlety"]) for r in results], f, indent=2)

    try:
        for domain, kind, subtlety in slots[len(results):]:
            if kind == "benign":
                session = generate_benign_session(domain)
            else:
                session = generate_hijacked_session(domain, subtlety)
            results.append({"session": session, "subtlety": subtlety})
            _save()  # progress survives a mid-run failure (e.g. rate limit)
            print(f"  generated {len(results)}/{len(slots)}: {domain} / {kind} / {subtlety}", flush=True)
    except Exception as e:
        print(f"Stopped after {len(results)}/{len(slots)} ({type(e).__name__}: {e}) — progress saved to {CACHE_PATH}.", flush=True)
        raise

    return results


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    force = "--force" in sys.argv
    dataset = generate_dataset(n_per_slot=n, force_regenerate=force)
    print(f"Generated/loaded {len(dataset)} sessions -> {CACHE_PATH}")
