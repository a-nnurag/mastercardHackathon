"""
SentenceTransformerEmbedder — wraps all-MiniLM-L6-v2, the same model
already used and verified working in defend/divergence.py. No new model
download; consistent embedding space across the project.
"""

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


class SentenceTransformerEmbedder:
    DIMENSION = 384

    def embed_text(self, text: str) -> list[float]:
        return _get_model().encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return _get_model().encode(texts).tolist()
