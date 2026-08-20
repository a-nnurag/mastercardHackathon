"""LangGraph state for the discovery workflow."""

from typing import TypedDict


class DiscoveryState(TypedDict):
    chunks: list[dict]        # remaining {"text", "source"} to process
    taxonomy: list[dict]      # accumulated AttackVector dicts, keyed separately by id in nodes.py
    _candidates: list         # transient: this round's extracted candidates, consumed by score_and_merge
