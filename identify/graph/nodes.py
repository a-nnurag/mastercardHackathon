"""
LangGraph node functions for the discovery workflow. Built as a factory
(`build_nodes`) closing over the shared LLM adapter, vector store,
knowledge graph, and embedder — these are real stateful resources
(an open Qdrant handle, an in-memory graph, a loaded embedding model)
that shouldn't be reconstructed on every node call.
"""

from functools import partial

from generate.llm_adapter import LLMAdapter

# Flush immediately -- this loop runs one LLM call per corpus chunk
# (dozens of them, ~15-20 min via local Ollama) and someone running it
# themselves needs live feedback, not output that appears in a burst at
# the end if stdout happens to be buffered.
print = partial(print, flush=True)
from identify.ingestion.embedder import SentenceTransformerEmbedder
from identify.knowledge_graph import AttackKnowledgeGraph
from identify.novelty_score import find_duplicate
from identify.schema import AttackVector
from identify.vectorstore.qdrant_store import QdrantStore

_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "attacks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "mechanism": {"type": "string"},
                    "defend_signal": {"type": "string"},
                    "category": {"type": "string"},
                    "delivery_vector": {"type": "string"},
                    "mechanism_type": {"type": "string"},
                },
                "required": ["name", "mechanism", "defend_signal", "category", "delivery_vector", "mechanism_type"],
            },
        },
    },
    "required": ["attacks"],
}

_EXTRACTION_PROMPT = """You are a cybersecurity threat-intelligence analyst building an attack \
taxonomy for AI shopping agents and agentic payment protocols.

Read the following text (the primary passage, plus related context from other documents \
in the corpus) and extract zero or more DISTINCT, CONCRETE attack techniques it describes. \
If the text is just statistics, background, or doesn't describe a specific attack mechanism, \
return an empty list — do not invent an attack that isn't actually described.

For each real attack found, provide:
- name: short, specific (e.g. "Branded Whisper Attack", "CSS-hidden JSON-LD prompt injection")
- mechanism: 1-3 sentences on HOW it actually works
- defend_signal: what data/signal could plausibly detect it
- category: one or two words (e.g. "prompt injection", "merchant impersonation", "social engineering")
- delivery_vector: a short tag for how the payload reaches the target (e.g. "typosquat-url", \
"hidden-css-text", "json-ld-metadata", "poisoned-tool-output", "seo-poisoning", "impersonation")
- mechanism_type: a short tag for the underlying mechanism class (e.g. "pre-signature-corruption", \
"post-authorization-injection", "identity-spoofing", "context-window-manipulation", "social-engineering")

PRIMARY PASSAGE:
{primary}

RELATED CONTEXT FROM OTHER DOCUMENTS:
{context}
"""


def build_nodes(adapter: LLMAdapter, store: QdrantStore, kg: AttackKnowledgeGraph, embedder: SentenceTransformerEmbedder):
    existing: dict[str, AttackVector] = {}
    counter = {"n": 0}
    progress = {"total": None, "done": 0}

    def extract_from_next_chunk(state: dict) -> dict:
        if progress["total"] is None:
            progress["total"] = len(state["chunks"])  # set once, on the first call
        progress["done"] += 1

        chunk, *rest = state["chunks"]
        print(f"[chunk {progress['done']}/{progress['total']}] extracting from {chunk['source']}...")
        related = [h for h in store.similarity_search(chunk["text"], top_k=5) if h["text"] != chunk["text"]][:4]
        context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in related)

        prompt = _EXTRACTION_PROMPT.format(primary=chunk["text"], context=context or "(none)")
        try:
            result = adapter.generate_json(prompt, _EXTRACTION_SCHEMA)
        except Exception as e:
            print(f"  extraction failed for chunk from {chunk['source']}: {type(e).__name__}: {e}")
            result = {"attacks": []}

        candidates = []
        for raw in result.get("attacks", []):
            try:
                candidates.append(AttackVector(**raw, grounding=[chunk["source"]]))
            except TypeError:
                continue  # malformed candidate from the LLM — skip rather than crash the run

        return {"chunks": rest, "_candidates": candidates}

    def score_and_merge(state: dict) -> dict:
        taxonomy = list(state["taxonomy"])
        for candidate in state.get("_candidates", []):
            dup_id = find_duplicate(candidate, existing, kg, embedder)
            if dup_id is not None:
                source = candidate.grounding[0]
                if source not in existing[dup_id].grounding:
                    existing[dup_id].grounding.append(source)
                    kg.add_grounding(dup_id, source)
                    for i, entry in enumerate(taxonomy):
                        if entry["name"] == existing[dup_id].name:
                            taxonomy[i] = existing[dup_id].to_dict()
                print(f"  duplicate of existing entry: {existing[dup_id].name!r} <- {candidate.name!r}")
                continue

            counter["n"] += 1
            attack_id = f"attack_{counter['n']}"
            existing[attack_id] = candidate
            kg.add_attack(attack_id, candidate)
            taxonomy.append(candidate.to_dict())
            print(f"  new attack: {candidate.name!r} ({candidate.category})")

        return {"taxonomy": taxonomy}

    def has_more_chunks(state: dict) -> str:
        from langgraph.graph import END

        return "extract_from_next_chunk" if state["chunks"] else END

    return extract_from_next_chunk, score_and_merge, has_more_chunks
