import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identify.ingestion.embedder import SentenceTransformerEmbedder
from identify.novelty_score import consolidate_by_name

_EMBEDDER = SentenceTransformerEmbedder()


def _entry(name, mechanism, source="https://example.com/a"):
    return {
        "name": name, "mechanism": mechanism, "defend_signal": "x", "category": "y",
        "delivery_vector": "z", "mechanism_type": "w", "grounding": [source],
    }


def test_merges_same_name_different_paraphrase():
    entries = [
        _entry("Branded Whisper Attack", "Manipulates product ranking to favor attacker-controlled items.", "https://a.com"),
        _entry("Branded Whisper Attack", "Injects prompts that push malicious merchants to the top of results.", "https://b.com"),
    ]
    result = consolidate_by_name(entries, _EMBEDDER)
    assert len(result) == 1
    assert set(result[0]["grounding"]) == {"https://a.com", "https://b.com"}


def test_does_not_merge_similar_names_different_mechanisms():
    # Regression test: an earlier version merged these two purely on name
    # similarity (0.76) even though they're different real attacks — one
    # manipulates rankings, the other exfiltrates data. Requiring
    # mechanism-text similarity too prevents this false merge.
    entries = [
        _entry("Branded Whisper Attack", "Manipulates product ranking to favor attacker-controlled items.", "https://a.com"),
        _entry("Vault Whisper Attack", "Extracts sensitive user information by bypassing data protection mechanisms.", "https://b.com"),
    ]
    result = consolidate_by_name(entries, _EMBEDDER)
    assert len(result) == 2


def test_does_not_merge_unrelated_names():
    entries = [
        _entry("Branded Whisper Attack", "Manipulates product ranking.", "https://a.com"),
        _entry("Poisoned Tool Outputs", "Spoofs a price oracle to return an inflated price.", "https://b.com"),
    ]
    result = consolidate_by_name(entries, _EMBEDDER)
    assert len(result) == 2
