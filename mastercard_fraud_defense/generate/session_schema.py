"""
Agent-session schema.

IEEE-CIS has no concept of an AI shopping agent's session — it's just
transaction rows. This file defines the structure we invent and join onto
those rows. Locked per the plan: don't change field names after Day 2
without updating every downstream script that reads them.
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

    # --- computed signals (filled in by defend/divergence.py) ---
    utterance_artifact_divergence: Optional[float] = None   # headline feature
    constraint_drift: Optional[float] = None                # fallback primary signal
    ingestion_source_trust_score: Optional[float] = None    # fallback primary signal

    hops_since_intent: int = 0
    tool_calls_made: int = 0

    injection_present: bool = False    # ground-truth label, known only in synthetic data

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
            "utterance_artifact_divergence": self.utterance_artifact_divergence,
            "constraint_drift": self.constraint_drift,
            "ingestion_source_trust_score": self.ingestion_source_trust_score,
            "hops_since_intent": self.hops_since_intent,
            "tool_calls_made": self.tool_calls_made,
            "injection_present": self.injection_present,
        }
