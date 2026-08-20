"""
In-memory attack knowledge graph (networkx), hybridized with the Qdrant
vector store for novelty scoring in novelty_score.py.

Why a graph on top of embeddings, not instead of them: published work on
knowledge-graph-enhanced RAG for cyber threat intelligence (CyKG-RAG,
AgCyRAG, MITRE ATT&CK-to-CVE graphs) targets exactly this kind of task —
consolidating overlapping attack descriptions across many documents — and
GraphRAG-style approaches measurably beat vector-only RAG specifically on
cross-document aggregation/consolidation queries. Real prompt-injection
taxonomies (CrowdStrike's, academic surveys) also classify attacks along
multiple structured dimensions (delivery vector, mechanism/payload type)
that a graph represents naturally as shared nodes, where flat text fields
would only support similarity, not structure.

Kept as a single in-memory networkx.MultiDiGraph (no Neo4j/server) since
our corpus (~10 real documents, likely 10-30 discovered attacks) is far
smaller than the enterprise-scale examples in that research — the same
"no extra infra" principle already applied to Qdrant's local mode.
"""

import networkx as nx


class AttackKnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_attack(self, attack_id: str, attack: "AttackVector") -> None:
        self.graph.add_node(attack_id, type="attack", name=attack.name, category=attack.category)
        self._add_edge(attack_id, "delivery", attack.delivery_vector, "delivered_via")
        self._add_edge(attack_id, "mechanism_type", attack.mechanism_type, "has_mechanism")
        for source in attack.grounding:
            self._add_edge(attack_id, "source", source, "grounded_in")

    def add_grounding(self, attack_id: str, source: str) -> None:
        self._add_edge(attack_id, "source", source, "grounded_in")

    def _add_edge(self, attack_id: str, node_type: str, value: str, relation: str) -> None:
        node_id = f"{node_type}:{value}"
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, type=node_type, value=value)
        self.graph.add_edge(attack_id, node_id, relation=relation)

    def shares_delivery_and_mechanism(self, attack_id_a: str, delivery_vector: str, mechanism_type: str) -> bool:
        """True if an existing attack node shares BOTH the given delivery
        vector and mechanism type — used by novelty_score.py as
        corroborating structural evidence alongside text similarity."""
        neighbors = set(self.graph.successors(attack_id_a))
        return f"delivery:{delivery_vector}" in neighbors and f"mechanism_type:{mechanism_type}" in neighbors

    def attack_ids(self) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("type") == "attack"]

    def save(self, path: str) -> None:
        import json

        data = nx.node_link_data(self.graph, edges="edges")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
