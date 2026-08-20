import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identify.schema import AttackVector
from identify.knowledge_graph import AttackKnowledgeGraph


def _attack(name="Typosquat Checkout", delivery="typosquat-url", mechanism_type="pre-signature-corruption"):
    return AttackVector(
        name=name,
        mechanism="A lookalike domain is presented as the merchant checkout page.",
        defend_signal="ingestion_source_trust_score",
        category="merchant impersonation",
        delivery_vector=delivery,
        mechanism_type=mechanism_type,
        grounding=["https://example.com/report"],
    )


def test_attack_vector_round_trips_through_dict():
    attack = _attack()
    restored = AttackVector.from_dict(attack.to_dict())
    assert restored == attack


def test_knowledge_graph_adds_attack_and_edges():
    kg = AttackKnowledgeGraph()
    kg.add_attack("attack_1", _attack())

    assert "attack_1" in kg.attack_ids()
    assert kg.graph.has_edge("attack_1", "delivery:typosquat-url")
    assert kg.graph.has_edge("attack_1", "mechanism_type:pre-signature-corruption")
    assert kg.graph.has_edge("attack_1", "source:https://example.com/report")


def test_knowledge_graph_shares_delivery_and_mechanism():
    kg = AttackKnowledgeGraph()
    kg.add_attack("attack_1", _attack())

    assert kg.shares_delivery_and_mechanism("attack_1", "typosquat-url", "pre-signature-corruption")
    assert not kg.shares_delivery_and_mechanism("attack_1", "typosquat-url", "social-engineering")
    assert not kg.shares_delivery_and_mechanism("attack_1", "hidden-css-text", "pre-signature-corruption")


def test_knowledge_graph_add_grounding_appends_source_edge():
    kg = AttackKnowledgeGraph()
    kg.add_attack("attack_1", _attack())
    kg.add_grounding("attack_1", "https://example.com/second-report")

    assert kg.graph.has_edge("attack_1", "source:https://example.com/second-report")
