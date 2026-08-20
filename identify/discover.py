"""
Task 6 entry point: RAG retrieval -> LLM extraction -> structured attack
entry, orchestrated by a real LangGraph loop (identify/graph/workflow.py).

Defaults to OllamaAdapter for extraction: Groq's gpt-oss-120b/gpt-oss-20b
are both still at today's 200k-token daily cap from Task 2's generation
run, and this task alone would need dozens of extraction calls. Still
built against the same LLMAdapter interface everything else in this repo
uses — swapping back to Groq once quota resets is a one-line change here,
not a rewrite.

Produces identify/taxonomy.json (however many genuine, well-grounded
attacks the real corpus yields — not padded to any target count) and
identify/knowledge_graph.json (the accompanying attack/delivery/
mechanism/source graph).
"""

import json
import os

from generate.llm_adapter import OllamaAdapter
from identify.graph.workflow import build_workflow
from identify.ingestion.chunker import Chunker
from identify.ingestion.embedder import SentenceTransformerEmbedder
from identify.ingestion.loader import load_corpus_documents, load_injection_payload_samples
from identify.knowledge_graph import AttackKnowledgeGraph
from identify.novelty_score import consolidate_by_name
from identify.vectorstore.qdrant_store import QdrantStore

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_QDRANT_PATH = os.path.join(_BASE_DIR, "qdrant_data")
TAXONOMY_PATH = os.path.join(_BASE_DIR, "taxonomy.json")
KNOWLEDGE_GRAPH_PATH = os.path.join(_BASE_DIR, "knowledge_graph.json")


def _build_chunks() -> list[dict]:
    chunker = Chunker()
    chunks = []
    for doc in load_corpus_documents() + load_injection_payload_samples():
        for piece in chunker.split(doc["text"]):
            chunks.append({"text": piece, "source": doc["source"]})
    return chunks


def run(adapter=None) -> list[dict]:
    adapter = adapter or OllamaAdapter()
    embedder = SentenceTransformerEmbedder()

    chunks = _build_chunks()
    print(f"Corpus: {len(chunks)} chunks")

    store = QdrantStore(_QDRANT_PATH, embedder)
    store.add_documents([c["text"] for c in chunks], [c["source"] for c in chunks])
    print(f"Ingested into Qdrant: {store.count()} vectors")

    kg = AttackKnowledgeGraph()
    app = build_workflow(adapter, store, kg, embedder)

    result = app.invoke(
        {"chunks": chunks, "taxonomy": [], "_candidates": []},
        config={"recursion_limit": len(chunks) * 3 + 10},
    )

    raw_taxonomy = result["taxonomy"]
    consolidated = consolidate_by_name(raw_taxonomy, embedder)

    with open(TAXONOMY_PATH, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)
    kg.save(KNOWLEDGE_GRAPH_PATH)  # raw discovery trace, not consolidated — see novelty_score.py

    print(f"\nRaw discovery: {len(raw_taxonomy)} entries before name-based consolidation")
    print(f"Consolidated: {len(consolidated)} attack entries -> {TAXONOMY_PATH}")
    print(f"Knowledge graph (raw trace) -> {KNOWLEDGE_GRAPH_PATH} ({kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges)")
    return consolidated


if __name__ == "__main__":
    run()
