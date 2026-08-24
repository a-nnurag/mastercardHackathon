"""
Agent-session schema.

IEEE-CIS has no concept of an AI shopping agent's session — it's just
transaction rows. This file defines the raw session structure joined onto
those rows.

Invariants:
- AgentSession contains ONLY raw facts + evaluation ground truth (injection_present).
- Detector-derived outputs (divergence, drift, trust scores, etc.) belong to
  defend/contracts.py (SignalResult, SignalSet), not this raw input schema.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class MandateScope:
    categories: list[str]
    amount_cap: float
    merchant_allowlist: list[str]
    expiry: datetime


@dataclass
class AgentSession:
    agent_id: str
    agent_registry_status: str  # mirrors Agentic Token validity: "valid" | "revoked" | "expired"

    mandate_scope: MandateScope

    intent_artifact_hash: str          # mirrors Verifiable Intent's signed artifact
    raw_utterance: str                 # what the human actually said — ephemeral, not persisted in prod
    signed_artifact_text: str          # what got encoded into the signed mandate

    task_origin_url: str
    content_sources_ingested: list[str]

    hops_since_intent: int = 0
    tool_calls_made: int = 0

    injection_present: bool = False    # ground-truth label, known only in synthetic data

    # LLM-generated hidden attacker text for hijacked sessions (Task 2).
    # Present only on hijacked sessions, None for benign sessions.
    injection_payload_text: Optional[str] = None

    def to_row(self) -> dict:
        """Flatten to a dict so it can be joined onto an IEEE-CIS transaction row."""
        return {
            "agent_id": self.agent_id,
            "agent_registry_status": self.agent_registry_status,
            "mandate_categories": self.mandate_scope.categories,
            "mandate_amount_cap": self.mandate_scope.amount_cap,
            "mandate_merchant_allowlist": self.mandate_scope.merchant_allowlist,
            "task_origin_url": self.task_origin_url,
            "content_sources_ingested": self.content_sources_ingested,
            "hops_since_intent": self.hops_since_intent,
            "tool_calls_made": self.tool_calls_made,
            "injection_present": self.injection_present,
        }
