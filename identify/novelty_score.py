"""
Hybrid novelty scoring: text-embedding similarity + knowledge-graph
relationship overlap. Per plan.md's file layout (novelty_score.py:
"dedupe/rank new candidates vs. existing taxonomy").

Text similarity alone can't tell "same attack described twice" from
"related but genuinely distinct attack" — two entries can be worded very
differently yet share a delivery vector and mechanism type (real
duplicate), or worded similarly yet target a different mechanism
(distinct attack). Combining both signals is the actual point of a
hybrid graph+vector approach, not graph OR vector alone.
"""

from difflib import SequenceMatcher

import numpy as np

from identify.ingestion.embedder import SentenceTransformerEmbedder
from identify.knowledge_graph import AttackKnowledgeGraph
from identify.schema import AttackVector

DUPLICATE_THRESHOLD = 0.80  # cosine similarity on mechanism text considered "same attack"
NAME_SIMILARITY_THRESHOLD = 0.6   # see consolidate_by_name — requires BOTH this
MECHANISM_SIMILARITY_FLOOR = 0.45  # ...and this. Name alone false-merged "Branded Whisper
                                    # Attack" with "Vault Whisper Attack" (different real
                                    # attacks sharing a "Whisper Attack" suffix, name
                                    # similarity 0.76) — mechanism-text similarity is the
                                    # check that tells them apart.


def _cosine(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_duplicate(
    candidate: AttackVector,
    existing: dict[str, AttackVector],
    kg: AttackKnowledgeGraph,
    embedder: SentenceTransformerEmbedder,
) -> str | None:
    """Returns the attack_id of a matching existing entry if `candidate`
    is a duplicate/variant of it, else None (genuinely novel)."""
    if not existing:
        return None

    candidate_vec = embedder.embed_text(candidate.mechanism)
    best_id, best_sim = None, 0.0
    for attack_id, attack in existing.items():
        sim = _cosine(candidate_vec, embedder.embed_text(attack.mechanism))
        if sim > best_sim:
            best_id, best_sim = attack_id, sim

    if best_sim >= DUPLICATE_THRESHOLD:
        return best_id

    # Structural corroboration: high-but-not-quite-threshold text
    # similarity, PLUS sharing both delivery vector and mechanism type,
    # is stronger duplicate evidence than either signal alone.
    if best_sim >= DUPLICATE_THRESHOLD - 0.15 and best_id is not None:
        if kg.shares_delivery_and_mechanism(best_id, candidate.delivery_vector, candidate.mechanism_type):
            return best_id

    return None


def consolidate_by_name(taxonomy: list[dict], embedder: SentenceTransformerEmbedder) -> list[dict]:
    """
    Post-hoc consolidation pass, run once over the full discovered
    taxonomy after the graph finishes — NOT a replacement for
    find_duplicate(), a backstop for what it misses.

    In practice, a small local model (Ollama qwen2.5:3b-instruct, used
    for extraction here to avoid Groq's exhausted daily quota) doesn't
    tag delivery_vector/mechanism_type consistently across repeated
    mentions of the same real attack — e.g. "Branded Whisper Attack" got
    extracted from several chunks with different delivery_vector tags
    each time, so the KG-hybrid check in find_duplicate() never fired
    even though the *name* was identical every time.

    Name similarity alone is NOT enough, though — an earlier version of
    this function used name similarity only and incorrectly merged
    "Branded Whisper Attack" with "Vault Whisper Attack" (two genuinely
    different real attacks from the same paper — one manipulates product
    ranking, the other exfiltrates data — that happen to share a
    "Whisper Attack" name suffix, name similarity 0.76). Requiring BOTH
    name similarity AND mechanism-text similarity above a floor avoids
    that false merge while still catching same-name duplicates whose
    mechanism text was paraphrased differently across chunks.

    This does not fabricate anything — it only merges entries that were
    already independently discovered, keeping the richer (more-grounded)
    merged result instead of reporting redundant near-duplicates as if
    they were distinct attacks.
    """
    clusters: list[dict] = []
    cluster_vecs: list[list[float]] = []
    for entry in taxonomy:
        entry_vec = embedder.embed_text(entry["mechanism"])
        match_idx = next(
            (
                i for i, c in enumerate(clusters)
                if SequenceMatcher(None, c["name"].lower(), entry["name"].lower()).ratio() >= NAME_SIMILARITY_THRESHOLD
                and _cosine(entry_vec, cluster_vecs[i]) >= MECHANISM_SIMILARITY_FLOOR
            ),
            None,
        )
        if match_idx is None:
            clusters.append(dict(entry))
            cluster_vecs.append(entry_vec)
        else:
            match = clusters[match_idx]
            for source in entry["grounding"]:
                if source not in match["grounding"]:
                    match["grounding"].append(source)
    return clusters
