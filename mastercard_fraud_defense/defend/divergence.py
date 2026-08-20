"""
utterance_artifact_divergence — the headline feature.

Measures how different the signed artifact text is from what the human
actually said. Large divergence = possible hijack.

Two signals, both computed when possible:

  LEXICAL divergence (TF-IDF + cosine distance): measures *word overlap*.
  Catches topic-level shifts (travel -> electronics) because vocabulary
  barely overlaps. Weak spot: ordinary paraphrasing also scores high here,
  so it can't distinguish "benign rewording" from "subtle hijack" — both
  look like moderate-to-high divergence to a word-overlap metric.

  SEMANTIC divergence (sentence embeddings + cosine distance): measures
  *meaning*. Two sentences with almost no shared words ("book a flight" /
  "reserve air travel") score as near-identical here, which is exactly
  what lexical divergence gets wrong. This is the metric that should
  actually decide the Day 2 gate — lexical is a fast, always-available
  backstop, not the primary signal.

Semantic divergence needs a local sentence-transformers model (downloads
weights from huggingface.co on first run) or an embeddings API call
(OpenAI/Anthropic — needs an API key). Neither is reachable from this
chat's sandbox, so this file is written to detect that and fall back to
lexical-only with a clear warning, rather than crashing. In Claude Code on
your own machine, the semantic path should load normally the first time
you run it (it'll download ~90MB once, then cache).
"""

import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SEMANTIC_MODEL = None
_SEMANTIC_LOAD_ATTEMPTED = False
_SEMANTIC_LOAD_ERROR = None


def _get_semantic_model():
    """Lazy-load so importing this file never fails, even with no network."""
    global _SEMANTIC_MODEL, _SEMANTIC_LOAD_ATTEMPTED, _SEMANTIC_LOAD_ERROR
    if _SEMANTIC_LOAD_ATTEMPTED:
        return _SEMANTIC_MODEL
    _SEMANTIC_LOAD_ATTEMPTED = True
    try:
        from sentence_transformers import SentenceTransformer
        _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        _SEMANTIC_LOAD_ERROR = str(e)
        warnings.warn(
            f"Semantic model unavailable ({type(e).__name__}). "
            f"Falling back to lexical-only divergence. "
            f"This is expected in a no-network sandbox; check your "
            f"environment if this happens in Claude Code."
        )
    return _SEMANTIC_MODEL


def compute_lexical_divergence(utterance: str, artifact_text: str) -> float:
    """0 = identical wording, 1 = no word overlap at all."""
    vectorizer = TfidfVectorizer().fit([utterance, artifact_text])
    vecs = vectorizer.transform([utterance, artifact_text])
    similarity = cosine_similarity(vecs[0], vecs[1])[0][0]
    return 1.0 - similarity


def compute_semantic_divergence(utterance: str, artifact_text: str) -> float | None:
    """0 = same meaning, 1 = unrelated meaning. Returns None if model unavailable."""
    model = _get_semantic_model()
    if model is None:
        return None
    embeddings = model.encode([utterance, artifact_text])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return 1.0 - float(similarity)


def compute_divergence(utterance: str, artifact_text: str, weights=(0.7, 0.3)) -> float:
    """
    Combined divergence score used by the rest of the pipeline.
    Weighted toward semantic (0.7) when available, since that's the signal
    that should actually decide hijack-or-not. Falls back to pure lexical
    (weight 1.0) if no semantic model is loaded.
    """
    lexical = compute_lexical_divergence(utterance, artifact_text)
    semantic = compute_semantic_divergence(utterance, artifact_text)
    if semantic is None:
        return lexical
    w_sem, w_lex = weights
    return w_sem * semantic + w_lex * lexical


def score_sessions(sessions: list) -> list:
    """
    Fills in .utterance_artifact_divergence (combined score) on each session,
    plus keeps the two raw components around for inspection/comparison.
    """
    for s in sessions:
        lexical = compute_lexical_divergence(s.raw_utterance, s.signed_artifact_text)
        semantic = compute_semantic_divergence(s.raw_utterance, s.signed_artifact_text)
        s.lexical_divergence = lexical
        s.semantic_divergence = semantic
        s.utterance_artifact_divergence = (
            0.7 * semantic + 0.3 * lexical if semantic is not None else lexical
        )
    return sessions
