"""
Builds the compiled LangGraph: START -> extract_from_next_chunk ->
score_and_merge -> (more chunks? loop back to extract : END).

A real cyclic graph, not a linear 3-step wrapper — this is the part
D:\\rag's own README described as the architecture ("LangGraph Workflow")
but never actually implemented.
"""

from langgraph.graph import END, START, StateGraph

from generate.llm_adapter import LLMAdapter
from identify.graph.nodes import build_nodes
from identify.graph.state import DiscoveryState
from identify.ingestion.embedder import SentenceTransformerEmbedder
from identify.knowledge_graph import AttackKnowledgeGraph
from identify.vectorstore.qdrant_store import QdrantStore


def build_workflow(adapter: LLMAdapter, store: QdrantStore, kg: AttackKnowledgeGraph, embedder: SentenceTransformerEmbedder):
    extract_from_next_chunk, score_and_merge, has_more_chunks = build_nodes(adapter, store, kg, embedder)

    graph = StateGraph(DiscoveryState)
    graph.add_node("extract_from_next_chunk", extract_from_next_chunk)
    graph.add_node("score_and_merge", score_and_merge)

    graph.add_edge(START, "extract_from_next_chunk")
    graph.add_edge("extract_from_next_chunk", "score_and_merge")
    graph.add_conditional_edges("score_and_merge", has_more_chunks, {"extract_from_next_chunk": "extract_from_next_chunk", END: END})

    return graph.compile()
