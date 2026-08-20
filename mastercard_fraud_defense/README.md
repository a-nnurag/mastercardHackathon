# Mastercard Innovation Challenge — build in progress

## What's here right now

```
generate/session_schema.py       # locked agent-session data structure
generate/synthetic_sessions.py   # 5 legitimate + 5 hijacked hand-written examples
defend/divergence.py             # TF-IDF divergence scorer (TEMPORARY baseline)
defend/isolation_test.py         # Day 2 gate: does divergence separate the two groups?
```

Run the test:
```
pip install -r requirements.txt
python3 defend/isolation_test.py
```

## Current result (as of last run)

AUC = 1.000 on hand-written synthetic examples — but this is not yet a
trustworthy signal. See the docstring in `defend/divergence.py` for why:
TF-IDF catches obvious vocabulary shifts, not subtle real attacks, and the
legitimate-session mean divergence (0.668) is already uncomfortably close
to the range a subtle real attack would likely land in.

## Next steps, in order

1. **Swap the divergence scorer to a real embedding model.**
   Options: OpenAI/Anthropic embeddings API (needs an API key + network
   access), or a local `sentence-transformers` model (needs the model
   weights, which this sandbox can't download — do this on your own
   machine/cloud, not in this chat's environment).
   Only change needed: replace the body of `compute_divergence()` in
   `defend/divergence.py`. The rest of the pipeline (schema, test runner)
   doesn't need to change.

2. **Replace hand-written sessions with the real narrative generator.**
   Build `generate/narrative_generator.py`: an LLM prompted to produce
   (a) benign agent tasks and (b) injection payloads that shift the signed
   artifact subtly — not just swapping to a totally unrelated category.
   Rerun `isolation_test.py` against that output. THIS result is the real
   Day 2 gate, not the one above.

3. **If the real test doesn't clear ~0.85+ AUC with realistic separation
   margins:** pivot immediately to `constraint_drift` +
   `ingestion_source_trust_score` as the primary signal, per the plan's
   kill criteria. Don't force the divergence feature if it doesn't hold up.

4. Once the signal is confirmed: join `generate/session_schema.py` rows
   onto real IEEE-CIS transaction data (Day-3/4 work) and start the
   LightGBM baseline (Day 5 hard gate).

## Not built yet
- `identify/` — RAG taxonomy agent (empty, reuses your `production_rag`)
- `generate/narrative_generator.py`
- `generate/ctgan_pipeline.py`
- `defend/rules.py`, `defend/lightgbm_baseline.py`, `defend/gnn.py`, `defend/llm_verdict.py`
- `mutator/` — confidence-guided mutation loop
