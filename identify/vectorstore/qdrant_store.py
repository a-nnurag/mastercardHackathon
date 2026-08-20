"""
QdrantStore — local/in-process Qdrant (on-disk, no server), same role as
D:\\rag's chroma_client.py but for Qdrant. Fresh code.
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from identify.ingestion.embedder import SentenceTransformerEmbedder

COLLECTION_NAME = "identify_corpus"


class QdrantStore:
    def __init__(self, path: str, embedder: SentenceTransformerEmbedder):
        self._client = QdrantClient(path=path)
        self._embedder = embedder
        if not self._client.collection_exists(COLLECTION_NAME):
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=embedder.DIMENSION, distance=Distance.COSINE),
            )

    def add_documents(self, chunks: list[str], sources: list[str]) -> None:
        vectors = self._embedder.embed_batch(chunks)
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": chunk, "source": source})
            for chunk, source, vector in zip(chunks, sources, vectors)
        ]
        self._client.upsert(collection_name=COLLECTION_NAME, points=points)

    def similarity_search(self, query_text: str, top_k: int = 4) -> list[dict]:
        query_vector = self._embedder.embed_text(query_text)
        hits = self._client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=top_k).points
        return [{"text": h.payload["text"], "source": h.payload["source"], "score": h.score} for h in hits]

    def count(self) -> int:
        return self._client.count(COLLECTION_NAME).count
