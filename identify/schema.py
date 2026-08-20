"""
AttackVector — the structured entry discover.py extracts from the corpus.

Matches TEAM_BRIEF.md Part 10's actual table columns (name, mechanism,
grounding, defend signal) plus two content-derived tags (delivery_vector,
mechanism_type) that knowledge_graph.py uses as shared graph nodes for
hybrid novelty scoring. No `region` field: most real sources don't state
one per attack, and Part 10's actual table doesn't have that column
either, despite the surrounding prose mentioning "5 regions" for the
full aspirational 48-entry taxonomy — not fabricating a field just to
match that number.
"""

from dataclasses import dataclass, field


@dataclass
class AttackVector:
    name: str
    mechanism: str
    defend_signal: str
    category: str            # LLM-assigned from real content (e.g. "prompt injection")
    delivery_vector: str     # short tag, e.g. "typosquat-url", "hidden-css-text"
    mechanism_type: str      # short tag, e.g. "pre-signature-corruption"
    grounding: list[str] = field(default_factory=list)  # source URLs; can accumulate
    # Task 10 (Mutator): candidates the Mutator writes back from Defend's
    # misses are unreviewed by design (a human hasn't vetted them yet) —
    # additive field, defaults True so the 61 already-discovered entries
    # (found before this field existed) round-trip unchanged.
    reviewed: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mechanism": self.mechanism,
            "defend_signal": self.defend_signal,
            "category": self.category,
            "delivery_vector": self.delivery_vector,
            "mechanism_type": self.mechanism_type,
            "grounding": self.grounding,
            "reviewed": self.reviewed,
        }

    @staticmethod
    def from_dict(d: dict) -> "AttackVector":
        return AttackVector(
            name=d["name"],
            mechanism=d["mechanism"],
            defend_signal=d["defend_signal"],
            category=d["category"],
            delivery_vector=d["delivery_vector"],
            mechanism_type=d["mechanism_type"],
            grounding=list(d.get("grounding", [])),
            reviewed=d.get("reviewed", True),
        )
