# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Project context for Claude Code. Read this first — it's short on purpose
and just points you to the two longer docs, rather than repeating them.

## Commands

```bash
pip install -r requirements.txt

# Confirm the semantic embedding model actually loads before trusting any
# AUC number that depends on it (first run downloads ~90MB from huggingface.co):
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('OK')"

# The Day-2 kill-criterion gate — run this after any change to divergence.py,
# session_schema.py, or the session generator. Read the SEMANTIC/COMBINED AUC
# lines, not the lexical one (lexical-only is a known-weak fallback).
python3 defend/isolation_test.py
```

There is no test suite yet. Per `plan.md` §5, new modules get
`tests/test_<module>.py` run with `pytest`; run a single test the normal way
(`pytest tests/test_foo.py::test_bar`) once that directory exists.

IEEE-CIS dataset and API-key setup (needed starting at plan.md Task 2/4) are
listed in `plan.md` §3 — not needed for the current isolation-test work.

## Architecture

Three pillars plus a feedback loop (full diagram in `TEAM_BRIEF.md` §4.1):
**Identify** (RAG taxonomy agent, not built yet) feeds attack patterns to
**Generate** (produces `AgentSession` test data), which feeds **Defend**
(scores sessions for hijack risk), whose misses feed the **Mutator**
(not built yet) to produce harder Generate rounds and candidate new
Identify taxonomy entries. The novelty is the closed loop, not any single
component — see `plan.md` §4 for build order and per-task definitions of
done.

Everything is built around one locked data structure,
`generate/session_schema.py`'s `AgentSession` (+ nested `MandateScope`):
it represents an AI shopping agent's session — what the human asked for
(`raw_utterance`), what actually got cryptographically signed
(`signed_artifact_text`), and the agent's spending mandate. It has no
natural counterpart in the IEEE-CIS transaction dataset; the plan is to
generate these synthetically and join them onto real IEEE-CIS rows via
`AgentSession.to_row()`. Never change a field name/type here without
updating every downstream consumer — see the non-negotiable rules below.

The headline detection signal, `utterance_artifact_divergence`
(`defend/divergence.py`), measures how far `signed_artifact_text` drifted
from `raw_utterance` — the theory being that a prompt-injection hijack
happens *before* signing, so the signed artifact encodes the attacker's
goal while VI's downstream cart-vs-mandate check still passes cleanly
(this is the entire thesis in `TEAM_BRIEF.md` §1.2). It's computed two
ways — lexical (TF-IDF, always available) and semantic (sentence-transformers,
needs the model to load) — combined 0.7×semantic + 0.3×lexical. Lexical
catches topic swaps; semantic is the one that should decide the real gate,
since it also catches subtle same-domain hijacks that lexical misses. If
this signal doesn't separate hijacked from legitimate sessions at
`defend/isolation_test.py`'s gate, the plan's fallback primary signal is
`constraint_drift` + `ingestion_source_trust_score` (mandate-scope rule
checks + URL provenance) — see kill criteria in `TEAM_BRIEF.md` Part 8 and
`plan.md` §8.

## task.md

`task.md` is a running log, not a spec — update it after finishing each
task from `plan.md` (what was done, the actual measured result, what's
next). Keep `plan.md`/`TEAM_BRIEF.md` as the source of truth for what
*should* happen; `task.md` records what *did* happen and when.

## A second copy of this project exists — don't edit it

`mastercard_fraud_defense/` (nested inside this repo) is a frozen snapshot
of an earlier deliverable milestone — not a working directory, not a
packaging target, and not kept in sync going forward. All active
development happens at the repo root. Don't copy changes into it and
don't treat anything inside it as more current than the root.

## Read before writing any code

1. **`TEAM_BRIEF.md`** — positioning, why the system is architected this
   way, the demo script, judge Q&A, and the numeric kill-criteria
   thresholds (§4.7 fidelity validation, §4.8 metrics, Part 8 kill
   criteria, Part 10 taxonomy entries). This is the *why* and *what the
   thresholds are*.
2. **`plan.md`** — the engineering execution spec: file structure, task
   order, function signatures, and a definition-of-done per task. This is
   the *what to build next and how to know it's done*. Work through it
   task by task; don't skip ahead.

## If the two docs ever disagree

`plan.md` is more current on implementation details (file layout,
function signatures, task sequencing) — it was written after
`TEAM_BRIEF.md` and reflects what's actually in the repo. `TEAM_BRIEF.md`
is authoritative on positioning language, the demo script, and any numeric
threshold it states explicitly (e.g. the LightGBM gate, AUC targets). If
you hit a real conflict — not just a rewording, an actual contradiction on
a threshold or scope decision — stop and ask rather than picking one
silently.

## Current repo state

`plan.md` §2 has the up-to-date list of what's built vs. placeholder.
Check it before touching `generate/session_schema.py`,
`defend/divergence.py`, or `defend/isolation_test.py` — they already
exist and their status (done / placeholder / unverified) is documented
there. Don't regenerate them from scratch without checking §2 first.

## Non-negotiable rules (also in plan.md §8, repeated here because they matter most)

- Don't change any field name or type in `AgentSession` or `MandateScope` — the schema is locked. If a change genuinely seems necessary, stop and ask.
- Don't substitute a different base dataset for IEEE-CIS.
- Don't call a kill-criterion gate "basically passed" when the measured number is below the stated threshold. Report the real number.
- Don't cut one of the two attack types or add a third without asking.
- Don't blur the line between "built" and "documented only" in any taxonomy or deck language — that distinction was deliberately fought for and matters for submission honesty.
- Don't fabricate a metric. If a number isn't computed, don't print something that looks computed.

## Where things are

```
TEAM_BRIEF.md          — positioning, demo script, judge Q&A, kill criteria thresholds
plan.md                — engineering task list, in build order
README.md              — quick status snapshot, keep updated as tasks complete
generate/               — session schema, synthetic/generated sessions, (later) IEEE-CIS join
defend/                 — divergence scoring, isolation test, (later) rules/LightGBM/GNN/verdict
identify/                — (empty — task 6 in plan.md)
data/                    — IEEE-CIS goes here once downloaded (see plan.md §3)
```
