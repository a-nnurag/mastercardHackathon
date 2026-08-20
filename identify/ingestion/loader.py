"""
Loads the corpus: fetched threat-report text files plus a sample of our
own Task 2 injection_payload_text values. Mirrors D:\\rag's PDFLoader
pattern (a plain .load(path) -> text interface) — fresh code, not
imported from that project.
"""

import os

_CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus")


class TextFileLoader:
    def load(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


def load_corpus_documents() -> list[dict]:
    """Returns [{"text": ..., "source": ...}] for every corpus/*.txt file.
    `source` is pulled from the file's own "SOURCE: <url>" header line."""
    loader = TextFileLoader()
    documents = []
    for filename in sorted(os.listdir(_CORPUS_DIR)):
        if not filename.endswith(".txt"):
            continue
        text = loader.load(os.path.join(_CORPUS_DIR, filename))
        first_line = text.splitlines()[0] if text else ""
        source = first_line.removeprefix("SOURCE: ").strip() if first_line.startswith("SOURCE:") else filename
        documents.append({"text": text, "source": source})
    return documents


def load_injection_payload_samples(n_per_domain: int = 2) -> list[dict]:
    """A representative sample of real injection_payload_text values from
    Task 2's generated sessions (not all 140 hijacked ones — that would
    just be near-duplicate noise the novelty scorer would collapse
    anyway). Per plan.md's explicit instruction to feed these into
    Identify as examples."""
    from generate.generated_sessions import load_cached_dataset

    dataset = load_cached_dataset()
    by_domain_subtlety = {}
    for d in dataset:
        session, subtlety = d["session"], d["subtlety"]
        if not session.injection_payload_text:
            continue
        key = (session.mandate_scope.categories[0], subtlety)
        by_domain_subtlety.setdefault(key, []).append(session)

    documents = []
    for (domain, subtlety), sessions in by_domain_subtlety.items():
        for session in sessions[:n_per_domain]:
            documents.append({
                "text": (
                    f"Domain: {domain}. Subtlety: {subtlety}. "
                    f"Injection payload (synthetic, LLM-generated for this project's own "
                    f"red-team dataset): {session.injection_payload_text}"
                ),
                "source": f"synthetic:generated_sessions.json:{session.agent_id}",
            })
    return documents
