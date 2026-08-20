# Mastercard Innovation Challenge — build in progress

**Status (2026-08-20): Tasks 1-10 of `plan.md`'s 12-task build order are
done** — the full three-pillar system (Identify/Generate/Defend) plus
the closed-loop Mutator, all with real measured results, several real
bugs found and fixed along the way (see `task.md` for the full log).
**Tasks 11 (web dashboard) and 12 (integration contract + latency
measurement) are not built yet.** Don't read "10 of 12 done" as "done" —
per this repo's own rules, that distinction matters.

## What's here right now

```
generate/session_schema.py       # locked agent-session data structure
generate/synthetic_sessions.py   # 5 legitimate + 5 hijacked hand-written examples
generate/llm_adapter.py          # LLMAdapter interface + Ollama/Groq/Gemini implementations
generate/narrative_generator.py  # real LLM-generated sessions (obvious/subtle hijacks)
generate/generated_sessions.py   # orchestrates + caches generated_sessions.json
defend/divergence.py             # semantic+lexical divergence scorer (headline signal — see finding below)
defend/constraint_drift.py       # fallback signal: mandate-scope rule checks
defend/isolation_test.py         # Day 2 gate: hand-written set, divergence only
defend/isolation_test_generated.py  # same gate, real LLM-generated set, obvious/subtle split
defend/rules.py                  # Defend layer 1: hard mandate-scope checks, no ML
generate/join_ieee_cis.py        # joins generated sessions onto real IEEE-CIS transaction rows
data/ieee-fraud-detection/       # real IEEE-CIS train_transaction.csv + train_identity.csv (not committed)
data/joined_sessions.csv         # 210 sessions x 448 columns, output of the join above
defend/lightgbm_baseline.py      # feature engineering + single full-data fit (importance only)
defend/evaluation.py             # THE HARD GATE: stratified 5-fold CV, real held-out metrics
identify/corpus/                 # 9 real, individually-fetched threat-report documents
identify/discover.py             # entry point: RAG + LangGraph loop -> taxonomy.json
identify/graph/                  # LangGraph state/nodes/workflow (the real orchestration loop)
identify/knowledge_graph.py      # in-memory networkx graph, hybrid novelty scoring with Qdrant
identify/vectorstore/qdrant_store.py  # local/on-disk Qdrant wrapper
identify/taxonomy.json           # output: real discovered attack entries
identify/knowledge_graph.json    # output: raw discovery-trace graph
defend/content_layer.py          # Defend layer 3: content/semantic injection scoring (Jaccard)
defend/content_layer_eval.py     # precision/recall + confidence-calibration (Brier/ECE)
generate/merchant_network.py     # attack #2: synthetic shell-merchant network, real domains
defend/gnn.py                    # Defend layer 4: GraphSAGE over merchant/acquirer/beneficiary
data/merchant_network.json       # output: the merchant graph (node-link format)
defend/llm_verdict.py            # Defend layer 5: fuses layers 1-4 into a plain-English verdict
mutator/mutate.py                # closed loop: hardens Defend's misses, writes back to Identify
```

Run the tests:
```
pip install -r requirements.txt
python3 defend/isolation_test.py
python3 -m pytest tests/ -v
```

## Task 2 complete (2026-08-20): headline signal fails on subtle attacks — primary signal pivoted

Full result on the complete 210-session real LLM-generated set (70 legit
/ 70 obvious-hijack / 70 subtle-hijack, generated locally via Ollama —
see `task.md` for the three-provider journey to get here). Reproduce with
`python3 defend/isolation_test_generated.py` (reads the existing cache,
no API calls):

```
utterance_artifact_divergence:  OBVIOUS AUC=1.000   SUBTLE AUC=0.419  (barely above chance)
constraint_drift:               OBVIOUS AUC=1.000   SUBTLE AUC=0.979
ingestion_source_trust_score:   OBVIOUS AUC=0.979   SUBTLE AUC=0.905
```

The headline `utterance_artifact_divergence` signal separates obvious
topic-swap hijacks perfectly, but on subtle same-domain hijacks (only
amount/merchant/destination shifted — the realistic attack) it stays well
below a useful threshold: keeping the wording close, by design, makes a
subtle hijack read as semantically similar to the original request. The
Task 3 fallback signals don't have this blind spot — they check the
mandate cap and merchant allowlist directly rather than semantic
similarity — and hold up strongly on both attack styles.

**Decision (2026-08-20, confirmed): `constraint_drift` +
`ingestion_source_trust_score` are now the primary Defend signal, not
`utterance_artifact_divergence`** — see the Decision Log at the top of
`task.md` for the full record of the question asked and answer given.
Divergence isn't abandoned (still catches obvious hijacks perfectly,
stays as a secondary signal) but no longer the headline claim. This
affects `TEAM_BRIEF.md` §4.3's framing of divergence as "the core novelty
claim" — flagged, not silently rewritten in the positioning doc.

## Superseded: hand-written-set result (2026-08-19)

Real semantic-model run (previous AUC=1.000 note below was lexical-only,
semantic model wasn't loaded in that sandbox — see `task.md` for the full
before/after):

```
SEMANTIC (embeddings):  legit mean=0.194 std=0.053   hijack mean=0.673 std=0.044   AUC = 1.000
COMBINED (pipeline):    legit mean=0.336 std=0.044   hijack mean=0.771 std=0.031   AUC = 1.000
```

Clears the Task 1 gate (≥0.85) with a clean margin on the 10 hand-written
examples. **This is not yet the real Day 2 gate result** — per `plan.md`
Task 2, hand-written hijack examples are known to be easier to separate
than realistic subtle attacks (same-domain, only amount/merchant shifted).
The trustworthy number comes once `generate/narrative_generator.py`
produces the real obvious/subtle test set — not built yet, blocked on an
`ANTHROPIC_API_KEY`/`OPENAI_API_KEY`.

Old lexical-only note, for reference: legitimate-session mean divergence
was 0.668 — uncomfortably close to where a subtle real attack might land.
TF-IDF catches obvious vocabulary shifts, not subtle real attacks; that's
exactly why the semantic path matters and this rerun was needed.

## Task 4 complete (2026-08-20): IEEE-CIS join + Rules layer

`generate/join_ieee_cis.py` joins the 210 generated sessions onto real
IEEE-CIS transaction rows (sampled only from `isFraud == 0` — a hijacked
agent payment is supposed to look like an *ordinary* transaction on the
card rail, so pairing with an already-fraud-flagged row would muddy that
thesis). Output: `data/joined_sessions.csv`. This is a **fabricated
pairing** — no real dataset links AI-agent sessions to IEEE-CIS
transactions — stated explicitly in the module docstring, not implied as
real.

`defend/rules.py` — Defend layer 1, hard mandate-scope checks, no ML.
Reproduce with `python3 -m defend.rules`:

| Subset | Flagged |
|---|---|
| benign (n=70) | 0/70 (zero false positives) |
| obvious-hijack (n=70) | 70/70 (100% recall) |
| subtle-hijack (n=70) | 67/70 (95.7% recall) |

## Task 5 complete (2026-08-20): LightGBM baseline — the hard gate, cleared

Reproduce with `python3 -m defend.evaluation` (installs as
`pip install lightgbm`, already in `requirements.txt`). Real result,
stratified 5-fold CV, out-of-fold predictions (n=210, 443 features):

| Subset | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Overall | 0.993 | 0.957 | 0.975 | 0.986 |
| Obvious vs benign | 0.986 | 1.000 | 0.993 | 1.000 |
| Subtle vs benign | 0.985 | 0.914 | 0.948 | 0.971 |

False-positive rate on legitimate traffic: **0.014 (1/70)** —
`TEAM_BRIEF.md` §4.8, reported as its own line, non-negotiable.

Feature importance (gain) is dominated by `constraint_drift` (~93% of
total gain), with `ingestion_source_trust_score` and
`utterance_artifact_divergence` also contributing and the real IEEE-CIS
transaction columns showing small but non-zero importance — direct
evidence for `TEAM_BRIEF.md` §1.2's thesis that classic transaction
features barely help here.

**The first run showed a suspicious AUC=1.000 everywhere** — traced to a
real label-leakage bug in Task 2's `narrative_generator.py`
(`hops_since_intent` was set to 0 for every benign session and a random
1-4 for every hijacked one, a deterministic encoding of the label itself,
not a narrative signal). Fixed at the source, the existing session cache
patched (no LLM re-spend needed — pure bookkeeping field), and a
regression test added. Numbers above are post-fix. Full story in
`task.md`. Small-sample caveat stands: n=210, 443 features is a real
small-n-large-p situation — a first honest read of feasibility, not a
bulletproof production number.

## Task 6 complete (2026-08-20): Identify RAG agent — LangGraph + Groq/Ollama + Qdrant + networkx

Built fresh (not imported from `D:\rag`, the `production_rag` candidate
plan.md pointed at — most of its files turned out to be empty scaffolds
that never actually implemented the LangGraph orchestration its own
README describes). Mirrors that project's loader→chunker→embedder→
vector-store pattern, with LangGraph's loop actually built this time,
Qdrant instead of Chroma (local/on-disk, no server), and our own
`generate/llm_adapter.py` reused as-is for the LLM (Groq's models were
still at their daily quota cap from Task 2, so this defaults to the
already-pulled local `OllamaAdapter` — a one-line swap back once quota
resets).

Corpus: 9 real, individually-verified threat-report documents (Greshake
et al., arXiv 2601.22569, CSA, 2x Halborn, Ravelin, Darwinium, Visa,
Zscaler — Mastercard's page wasn't fetchable, skipped rather than
fabricated) plus a sample of Task 2's real `injection_payload_text`
values. Added a lightweight in-memory knowledge graph
(`identify/knowledge_graph.py`, `networkx`, no server) for hybrid
novelty scoring alongside embedding similarity — based on real published
precedent (CyKG-RAG-style graph+vector RAG measurably beats vector-only
RAG on cross-document consolidation tasks, which is exactly what
building a taxonomy from many overlapping reports is).

Reproduce with `python3 -m identify.discover` (~15-17 min via local
Ollama). Real result:

- Raw discovery: 88 entries → **61 after consolidation**
  (`identify/taxonomy.json`).
- **32 grounded in a real external threat-report URL**; 29 grounded only
  in our own synthetic injection-payload samples.
- Knowledge graph: 207 nodes, 281 edges (`identify/knowledge_graph.json`).

No 46/48-item taxonomy markdown exists anywhere on this machine (checked
before building) — this is the real, unforced count a real corpus
yields, not padded to match a target. A real consolidation bug got
caught and fixed here too: an early version merged "Branded Whisper
Attack" with the genuinely different "Vault Whisper Attack" purely on
name-string similarity — fixed by requiring mechanism-text similarity
too, with a regression test. Known limitation, stated plainly: the small
local model (`qwen2.5:3b-instruct`) is noticeably less consistent than
the Groq-hosted models used earlier (category-tag formatting varies,
some entry names are awkward, extraction isn't fully reproducible
run-to-run) — the mechanism is sound, a larger model would clean up the
output quality once quota allows it. Full story in `task.md`.

## Task 7 complete (2026-08-20): content/semantic injection-detection layer

Per `plan.md`'s instruction, searched for the referenced
"notification-router Jaccard-similarity" prior work before building —
found a real project at `D:\orchestrate_learning\a\Message-Notification-Router`.
Read its actual algorithm (`router/evidence.py`'s token-set Jaccard
overlap) and confidence-calibration methodology
(`eval/metrics.py`'s Brier score / ECE, the real numbers its own
`EVALUATION.md` reports) — fresh implementation here, nothing imported.

Since `AgentSession` has no scraped-page-text field, this layer scores
`injection_payload_text` (hijacked) vs `raw_utterance` (benign) — stated
plainly as the real proxy for "ingested content" it is. Reference phrases
are real: 4 quotes from Task 6's fetched corpus + 20 held-out real
`injection_payload_text` values (2/domain), with the eval set built from
the *other* 120 hijacked + all 70 benign sessions (no overlap). Reproduce
with `python3 -m defend.content_layer_eval`:

```
Threshold=0.15: precision=0.983 recall=0.992 F1=0.988  (TP=119 FP=2 FN=1)
Confidence calibration: Brier score = 0.2485   ECE = 0.3705
```

**Worth surfacing, not smoothing over:** the raw score discriminates
excellently once thresholded, but is a poor *calibrated* confidence
(Brier 0.2485 is barely above the 0.25 "uninformative" baseline) —
verified the Brier/ECE formulas against known sanity cases before
trusting this rather than assuming a bug. This is safe to use as a flag
after thresholding, not safe to present to a risk analyst as a literal
probability without a proper calibration mapping (out of scope for this
task's definition of done).

## Task 8 complete (2026-08-20): GNN over merchant/acquirer/beneficiary graph — attack #2

First generated data for attack #2 (merchant-network laundering).
Grounded in real data per `TEAM_BRIEF.md` §2.1's stated connection
between the two attacks: the 212 real malicious/typosquat domains and 19
real legitimate domains already present in the 210 generated sessions
become the actual merchant nodes (not invented names). Reproduce with
`python3 -m generate.merchant_network` then `python3 -m defend.gnn`.

Two real issues caught and fixed before trusting the result: (1) hijacked
sessions sometimes reference the genuine origin domain alongside the
fake one, which silently miscounted 13/19 real merchants as malicious —
fixed with a regression test on the actual set overlap; (2) the first
graph design gave every legitimate merchant a fully unique Acquirer,
making "shared acquirer" a perfect (and unrealistic) fraud tell — fixed
by drawing legitimate merchants from a small shared pool of processors
(real small businesses do this) while keeping Beneficiary sharing
exclusive to fraud rings, matching `TEAM_BRIEF.md` Part 10 item #48's
framing of shared acquirer *and* beneficiary *and* rapid registration
together, not any single signal alone.

`torch_geometric` (2.8.0.post1) installed as a clean pure-Python wheel —
**the GNN path converged and was used; the LightGBM-on-graph-features
fallback wasn't needed.** Real result (transductive node classification,
held-out n=66):

```
precision=0.968  recall=1.000  F1=0.984  AUC=1.000
```

Honest caveat: shared-beneficiary is still a strong, near-deterministic
signal by construction — defensible (it's the real canonical AML "follow
the money" signal, and `TEAM_BRIEF.md` names it as a defining ring
characteristic), but this result reflects a ring that hasn't tried to
obscure ownership. A ring that does would be a harder, more realistic
case for the eventual Mutator (Task 10) to produce, not something forced
here artificially.

## Task 9 complete (2026-08-20): LLM verdict layer — shipped, SHAP fallback not needed

Fuses all four existing Defend layers (rules, LightGBM, content, GNN —
each reused as-is, nothing reimplemented) into one plain-English verdict
for a risk analyst. Reproduce with `python3 -m defend.llm_verdict`.

Set a concrete acceptance bar before deciding LLM-vs-fallback (per
`plan.md`: "never ship a half-working layer, pick the fallback
explicitly"): 12 representative sessions, checked programmatically for
whether the stated risk level is directionally consistent with the four
numeric signals. **Result: 12/12 on both attempts — shipped, the
`defend/shap_fallback.py` fallback wasn't needed.** Ran on `OllamaAdapter`
(Groq's still capped from Task 2 — confirmed by checking, not assumed).
Sample explanation, correctly grounded in the real inputs, nothing
invented: *"This transaction deviates from the user's request for a
prescription and instead orders a high-priced weight loss supplement.
The amount exceeds the 2000 rupee cap, and the domain
'medifit-deals-offer.com' is not in the merchant allowlist... the
transaction should be blocked."*

Two small, honestly-reported model-quality wrinkles (neither failed the
acceptance bar): one early explanation's closing sentence disagreed with
its own structured `recommendation` field (fixed with an explicit prompt
instruction); a rarer case afterward paired `risk_level=LOW` with
`recommendation=BLOCK` — erring cautious, not reckless, and consistent
with the small-local-model limitations Tasks 6-7 already found. Not
chased further with more prompt tuning — diminishing returns, didn't
change the ship decision.

## Next steps, in order

## Task 10 complete (2026-08-20): Mutator closed loop

Reused every existing building block (Task 5's CV methodology, Task 2's
session-construction helpers, Task 4's IEEE-CIS sampling, Tasks 1/3's
scoring) — the only new logic is the mutation prompt, the round loop,
and honest reporting. Checked what Task 5's model actually misses before
designing anything: 6 hard cases (false negatives + low-confidence
catches), all subtle hijacks. Reproduce with `python3 -m mutator.mutate`.

**Three real bugs hit and fixed** (this task exercised the pipeline in
new ways every prior task hadn't): (1) the first mutation prompt never
told the LLM the actual mandate cap, so "make it harder" produced a
*bigger* deviation instead of a smaller one; (2) the fix over-corrected,
producing sessions *under* the cap entirely — no longer real violations;
fixed by requiring the amount stay a genuine (just smaller) violation;
(3) a real, general gap in Task 5's original `_CATEGORICAL_COLUMNS` list
— it worked on the original 210 rows only because two sparse IEEE-CIS
identity columns happened to be all-NaN there, and broke the moment
fresh sampling hit real values. Fixed generally (cast any remaining
non-numeric column, not just a fixed list) rather than adding two more
names — and the first attempt at that fix silently didn't work either
(pandas 3.0's `StringDtype` isn't `== object`), caught by a real
regression test, not by assuming the fix worked.

**Real per-round result** (5-fold CV each round, same methodology as
Task 5):

| Round | n | Precision | Recall | F1 | FP-rate |
|---|---|---|---|---|---|
| 1 (baseline) | 210 | 0.993 | 0.957 | 0.975 | 0.014 |
| 2 (+12 harder) | 222 | 0.993 | 0.987 | 0.990 | 0.014 |
| 3 (+4 harder) | 226 | 0.981 | 0.987 | 0.984 | 0.043 |

**Honest read:** Round 1→2 is a real improvement (recall 0.957→0.987,
4 of 6 original hard cases stopped being hard at all). **Round 2→3
recall didn't improve, and the benign false-positive rate roughly
tripled** — reported as the actual finding, not smoothed into a curve
that isn't there, per `plan.md`'s kill criteria.

16 new `AttackVector` entries written to `identify/taxonomy.json` with
`reviewed=False` (taxonomy now 77 entries, 16 unreviewed) — the
`reviewed` field is a purely additive schema change, verified the
original 61 entries default to `True` rather than assumed.

## Task 11 — Web dashboard (complete)

`dashboard/` — FastAPI + plain HTML/CSS/JS (no build step). Three views:
Sessions (all 226 real sessions, client-side filters, precomputed
rules/constraint-drift/content/GNN signals), Session Detail (raw
utterance / signed artifact / hidden injection payload + a live
"Get AI Verdict" button calling `defend.llm_verdict.verdict_for_session()`),
Metrics (Task 5's CV metrics and Task 8's GNN metrics re-run at startup,
plus Task 10's 3-round hardening curve as an inline SVG chart). Every
number on the page is computed from the real pipeline, none hardcoded.

Run: `python3 -m uvicorn dashboard.main:app --host 127.0.0.1 --port 8000`,
then open `http://127.0.0.1:8000`. Verified end-to-end in an actual
browser, not just via curl — session list/filters, a hijacked session's
detail view, a real Ollama-generated verdict, and the metrics view's
chart all confirmed rendering correctly. Building this surfaced and fixed
two real bugs: `verdict_for_session()` couldn't resolve any of Task 10's
16 Mutator-hardened sessions (only checked the base 210), and the new
metrics endpoint crashed on its first call (`.sum()` on a plain list
where every other slice had a numpy/pandas array). `tests/test_dashboard_data_access.py`
(5 tests, runs the real LightGBM/GNN pipeline rather than mocking it).
Full suite: **56/56 passing.**

## Task 12 — Integration contract + latency measurement (complete)

`defend/integration_contract.py` + `INTEGRATION_CONTRACT.md`. Defines
the JSON payload a session-risk score would emit into a Decision
Intelligence-style consumer (rides along with the signed Intent
Artifact, at the agent-provider/PSP layer, since the network never sees
the raw utterance) — built from the four fast Defend layers only (rules,
LightGBM, content, GNN), Task 9's LLM verdict deliberately excluded from
this latency-critical path. Real measured latency, not an estimate:
**mean 9.08 ms, p95 10.59 ms** for one session's full payload
(pre-trained model held in memory, 50 timed runs). `tests/test_integration_contract.py`
validates the payload against its own JSON Schema with `jsonschema.validate()`.
Full suite: **61/61 passing.**

**All 12 tasks in `plan.md`'s build order are now complete**, each with
real measured results.

## Next steps

**Reflect the signal pivot in `TEAM_BRIEF.md`-facing language.** Decided
in code/eval terms since Task 1; the positioning doc's §4.3 "core
novelty claim" framing around `utterance_artifact_divergence` still
needs updating to match the confirmed `constraint_drift`/
`ingestion_source_trust_score` pivot — flagged repeatedly across this
log, still not done, not tied to any single numbered task so easy to
lose track of.

## Not built yet
- `generate/ctgan_pipeline.py` (mentioned in early planning, superseded by the LLM-generation approach actually used)
# mastercardHackathon
