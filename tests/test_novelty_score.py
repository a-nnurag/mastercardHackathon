import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identify.schema import AttackVector
from identify.knowledge_graph import AttackKnowledgeGraph
from identify.ingestion.embedder import SentenceTransformerEmbedder
from identify.novelty_score import find_duplicate

_EMBEDDER = SentenceTransformerEmbedder()


def _attack(mechanism, delivery="typosquat-url", mechanism_type="pre-signature-corruption"):
    return AttackVector(
        name="Existing Attack",
        mechanism=mechanism,
        defend_signal="ingestion_source_trust_score",
        category="merchant impersonation",
        delivery_vector=delivery,
        mechanism_type=mechanism_type,
        grounding=["https://example.com/a"],
    )


def test_no_existing_entries_means_novel():
    kg = AttackKnowledgeGraph()
    candidate = _attack("A lookalike domain replaces the real checkout page.")
    assert find_duplicate(candidate, {}, kg, _EMBEDDER) is None


def test_near_identical_mechanism_text_is_duplicate():
    kg = AttackKnowledgeGraph()
    existing_attack = _attack("A lookalike domain replaces the real checkout page to steal payment details.")
    kg.add_attack("attack_1", existing_attack)
    existing = {"attack_1": existing_attack}

    candidate = _attack("A lookalike domain replaces the real checkout page to steal payment details.")
    assert find_duplicate(candidate, existing, kg, _EMBEDDER) == "attack_1"


def test_unrelated_mechanism_is_novel():
    kg = AttackKnowledgeGraph()
    existing_attack = _attack("A lookalike domain replaces the real checkout page to steal payment details.")
    kg.add_attack("attack_1", existing_attack)
    existing = {"attack_1": existing_attack}

    candidate = _attack(
        "A poisoned price oracle returns an inflated price that the agent silently accepts.",
        delivery="poisoned-tool-output", mechanism_type="tool-output-spoofing",
    )
    assert find_duplicate(candidate, existing, kg, _EMBEDDER) is None
