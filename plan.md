# Build plan — Mastercard Innovation Challenge fraud-defense system

> **Audience for this doc:** Claude Code, picking up an existing partial repo.
> This is an engineering execution spec, not the pitch/positioning doc (that's
> `TEAM_BRIEF.md`, separate, for the deck — don't merge the two).
>
> **Ground rule for whoever/whatever executes this:** don't change the
> `AgentSession` schema, don't fabricate IEEE-CIS data, and don't skip a kill
> criterion to make progress look better. If a gate fails, report that
> honestly and follow the fallback listed for it — don't quietly patch
> around it. Flag any of the "decisions not to make unilaterally" (§8) back
> to the human instead of guessing.

---

## 1. Project goal (one paragraph, for context)

Build a closed-loop, three-pillar system (Identify / Generate / Defend, plus
a Mutator feedback loop) that detects **pre-signature Intent Artifact
corruption** — an AI payment agent hijacked by prompt injection *before* it
signs a payment mandate, producing a cryptographically valid but
intent-corrupted transaction — and **transaction laundering via a
fraudulent merchant network** on the same card rail. Full positioning and
demo script live in `TEAM_BRIEF.md`. This doc is about what code gets
written, in what order, with what acceptance criteria.

---

## 2. Current repo state — READ THIS FIRST, don't redo it

```
mastercard_fraud_defense/
├── README.md                          # status notes, keep updated as you go
├── requirements.txt                    # scikit-learn, numpy, pandas, sentence-transformers
├── generate/
│   ├── session_schema.py              # ✅ DONE — AgentSession, MandateScope dataclasses. LOCKED.
│   └── synthetic_sessions.py          # ⚠️ PLACEHOLDER — 5 legit + 5 hijacked hand-written
│                                          examples across 5 domains. Real narrative generator
│                                          (task 4 below) replaces this.
├── defend/
│   ├── divergence.py                  # ✅ DONE, but semantic path UNVERIFIED — computes both
│   │                                     lexical (TF-IDF) and semantic (sentence-transformers,
│   │                                     all-MiniLM-L6-v2) divergence, combines 0.7*semantic +
│   │                                     0.3*lexical. Semantic model couldn't load in the
│   │                                     sandbox that wrote this (no huggingface.co access) —
│   │                                     confirm it loads in this environment before trusting
│   │                                     any AUC number that depends on it.
│   └── isolation_test.py              # ✅ DONE — reports lexical / semantic / combined AUC
│                                          separately. Last real run (lexical-only, semantic
│                                          unavailable): AUC 1.000, but flagged as not
│                                          trustworthy — see §6 task 1.
└── identify/
    └── (empty — task 6 below)
```

**Do not treat the AUC=1.000 result already in the repo as a passed gate.**
It was run without the semantic model and on hand-written (not
LLM-generated) hijack examples — both of which make separation artificially
easy. Task 1 below is to get a real number.

---

## 3. Environment setup (do this before anything else)

- [ ] Python 3.11+, `pip install -r requirements.txt`
- [ ] Confirm semantic model loads: `python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('OK')"` — first run downloads ~90MB, needs network access to huggingface.co
- [ ] **IEEE-CIS Fraud Detection dataset** (Kaggle) — needs a Kaggle account + API token (`~/.kaggle/kaggle.json`). Download with `kaggle competitions download -c ieee-fraud-detection`, unzip into `data/ieee_cis/`. This is the real-transaction fidelity baseline — do not substitute a different dataset (PaySim/UPI was explicitly rejected earlier for rail mismatch, see `TEAM_BRIEF.md` §4.7).
- [ ] Anthropic or OpenAI API key, for the narrative generator (task 4) and eventually the LLM verdict layer (task 9). Store as env var (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`), never hardcoded.
- [ ] `pip install lightgbm torch torch_geometric` — needed later for tasks 5 and 8, not needed yet.

---

## 4. Build order — one task at a time, in this sequence

Each task lists: files touched, what to build, and a **definition of done**
(a concrete, runnable check — not "looks good"). Don't start task N+1 until
task N's definition of done passes, except where marked parallelizable.

### Task 1 — Confirm the real divergence signal (BLOCKS EVERYTHING ELSE)

**Files:** `defend/isolation_test.py` (read-only, already correct), your environment only.

Run `python3 defend/isolation_test.py` in an environment where the semantic
model actually loads. Read the **SEMANTIC** and **COMBINED** AUC lines, not
the lexical one.

**Definition of done:** one of two outcomes, both acceptable — this task is
about getting a true answer, not a good-looking one:
- Semantic/combined AUC ≥ 0.85 on the existing 10 hand-written examples →
  proceed to Task 2.
- Semantic/combined AUC < 0.85 → **stop, don't proceed to Task 2.** Instead
  jump to Task 3 (fallback signal) and report the failure plainly.

### Task 2 — Real narrative generator (replaces hand-written examples)

**Files:** new `generate/narrative_generator.py`

Build an LLM-driven generator that produces `AgentSession` objects (reuse
the schema in `session_schema.py` — do not redefine it) across two classes:

```python
def generate_benign_session(domain: str) -> AgentSession: ...
def generate_hijacked_session(domain: str, subtlety: str = "obvious") -> AgentSession: ...
```

- `domain`: rotate through at least 8-10 categories (travel, groceries,
  electronics, subscriptions, dining, utilities, insurance, education,
  home-goods, pharmacy) — variety matters, don't let the LLM default to the
  same 2-3 examples repeatedly.
- `subtlety`: `"obvious"` (topic changes completely, like the current
  hand-written hijack examples) vs `"subtle"` (same-ish domain, wording
  stays close, only amount/merchant/destination shifts — e.g. "book a
  flight to Bangalore" → artifact signs a flight to a different city at 3x
  the price, still "travel"). **The subtle case is the one that actually
  matters** — it's the realistic attack, and it's the case lexical
  divergence is known to miss (see `divergence.py` docstring).
- Prompt the LLM to write the injection payload itself too (the hidden
  text an attacker would plant) — store it on a new field,
  `injection_payload_text`, even though it's not consumed by anything yet.
  Identify's RAG agent (task 6) will eventually want these as examples.

Generate ~200 sessions (mix of benign/hijacked, mix of obvious/subtle,
spread across all domains). Replace `synthetic_sessions.py`'s role by
adding a `generate/generated_sessions.py` that calls this and caches
output to `data/generated_sessions.json` (don't regenerate on every test
run — LLM calls cost money and time).

**Definition of done:** `python3 defend/isolation_test.py` rerun against
the new 200-session set, **split into obvious vs subtle subsets**, reports
AUC separately for each subset. This is the real Day-2-equivalent number.
Report both numbers honestly, even if the subtle-subset AUC is much lower
than the obvious-subset one — that gap *is* the finding.

### Task 3 — Fallback signal (build this in parallel with Task 2, don't skip even if Task 1 passed)

**Files:** new `defend/constraint_drift.py`

Implement `constraint_drift` and `ingestion_source_trust_score` as
described in `TEAM_BRIEF.md` §4.3/§8:
- `constraint_drift`: does the signed artifact violate the *mandate_scope*
  fields already on the session (category not in `categories`, amount over
  `amount_cap`, merchant not in `merchant_allowlist`)? This is closer to a
  rules check than an ML signal — deliberately simple, low-risk.
- `ingestion_source_trust_score`: a score for each URL in
  `content_sources_ingested` — start with a simple heuristic (domain age
  proxy: does it look like a typosquat of a known merchant? use string
  edit-distance against an allowlist), not a full threat-intel lookup.

**Definition of done:** both functions run standalone with unit tests
covering at least one clear-violation and one clear-pass case each. This
exists so that if Task 1/2 shows the divergence signal doesn't separate
well on subtle attacks, there's already a working alternative to pivot to
— not a scramble.

### Task 4 — Join to IEEE-CIS, build the Rules layer

**Files:** new `defend/rules.py`, new `generate/join_ieee_cis.py`

- `join_ieee_cis.py`: loads the downloaded IEEE-CIS transaction data,
  joins `AgentSession.to_row()` output onto sampled real transaction rows
  (per `TEAM_BRIEF.md` §4.6 — this is fabricated pairing, not real paired
  data, since no such dataset exists; state that limitation in the code
  comments, don't silently imply real pairing).
- `rules.py`: implements Defend layer 1 — hard checks against
  `mandate_scope` (amount cap breach, merchant allowlist miss, category
  violation, `agent_registry_status != "valid"`). Returns a flag + reason
  string per session, no ML.

**Definition of done:** running rules.py against the full generated+joined
dataset produces a flag/reason for every row, and recall on the
hijacked-obvious subset is meaningfully non-zero (rules alone should catch
the blatant cases; if it catches zero, something's wired wrong).

### Task 5 — LightGBM baseline (THE HARD FLOOR — per `TEAM_BRIEF.md`, this gate cannot be skipped)

**Files:** new `defend/lightgbm_baseline.py`, new `defend/evaluation.py`

Train a LightGBM classifier on the joined dataset (agent-session fields +
IEEE-CIS transaction features). `evaluation.py` computes, **per attack
type, never blended**: precision, recall, F1, AUC, and false-positive rate
on legitimate traffic specifically (this last one is non-negotiable per
`TEAM_BRIEF.md` §4.8).

**Definition of done:** real precision/recall/F1/AUC numbers exist and are
printed/logged, computed on a held-out split, not training data. If this
task can't produce a working classifier: **stop all downstream feature
work** (GNN, LLM verdict) and report that the floor wasn't met — this
mirrors the plan's Aug-23-equivalent hard gate. Don't proceed past this
task silently on a broken baseline.

### Task 6 — Identify: RAG attack-discovery agent (mandatory pillar, not optional)

**Files:** new `identify/discover.py`, `identify/schema.py`, `identify/taxonomy.json`

Reuses the existing `production_rag` pipeline (already built prior to this
project — don't rebuild RAG from scratch, adapt the existing one). Corpus:
threat reports already gathered for `TEAM_BRIEF.md` (Mastercard, Visa,
Darwinium, Ravelin, Zscaler, arXiv 2601.22569, CSA, Halborn) plus, per
Task 2, the `injection_payload_text` examples generated there. Output:
structured `AttackVector` entries matching the schema already established
in the existing 48-item taxonomy markdown (don't invent a new schema —
match the existing one so `taxonomy.json` and the markdown doc stay
consistent).

**Definition of done:** running `discover.py` against the corpus produces
`taxonomy.json` with at least the same 48 entries as the existing markdown
taxonomy, each with the fields: attack name, mechanism, grounding source,
Defend signal. This is required for submission compliance — a markdown
document alone does not satisfy "Identify" as runnable code.

### Task 7 — Content/semantic injection-detection layer

**Files:** new `defend/content_layer.py`

Classifies whether page content ingested via `content_sources_ingested`
contains an injection payload (hidden CSS text, JSON-LD poisoning,
typosquat indicators). Reuse the confidence-calibration pattern from prior
work (referenced in `TEAM_BRIEF.md` — check existing codebase for the
notification-router Jaccard-similarity approach before building new).

**Definition of done:** standalone test distinguishing at least the
generated hijacked sessions' `content_sources_ingested` from the benign
ones, reported as precision/recall like the other layers.

### Task 8 — GNN merchant/acquirer/beneficiary layer (attack #2, optional — has a named fallback)

**Files:** new `defend/gnn.py`, new `generate/merchant_network.py`

Build the synthetic shell-merchant network (shared acquirer, shared
beneficial owner, rapid-registration domains — per `TEAM_BRIEF.md` Part 10
item #48). Train a GNN (PyTorch Geometric) over
merchant→acquirer→beneficiary edges to flag laundering clusters.

**Definition of done / fallback:** if unstable or not converging within
reasonable effort, per the plan's kill criteria: drop to graph *features*
(node degree, hop-depth, community ID) fed into the Task-5 LightGBM model
instead of a standalone GNN. Either outcome is acceptable — report which
one happened.

### Task 9 — LLM verdict layer (optional, has a named fallback)

**Files:** new `defend/llm_verdict.py`

Fuses outputs of layers 1-4 (rules, LightGBM, content, GNN) into a
plain-English explanation for a risk analyst.

**Definition of done / fallback:** if this doesn't come together cleanly,
fall back to a weighted ensemble of the four layer scores + SHAP feature
importances (`defend/shap_fallback.py`). Never ship a half-working LLM
verdict layer live — pick the fallback explicitly instead.

### Task 10 — Mutator: confidence-guided mutation loop

**Files:** new `mutator/mutate.py`

Takes false-negatives + low-confidence catches from Task 5's evaluation
output, feeds them back to Task 2's narrative generator with a prompt like
"here's a session that evaded detection, generate a harder variant that
preserves the evasion property." Also writes new-attack candidates back
into Task 6's `identify/taxonomy.json` as unreviewed entries (this is the
three-way closed loop described in the architecture diagram).

**Definition of done:** running two full mutation rounds and rerunning
Task 5's evaluation on each round produces a logged per-round
precision/recall/F1/FP-rate table. Report the round-over-round delta
honestly — if round 2 doesn't improve on round 1, say so; don't claim an
improvement curve that isn't there (per `TEAM_BRIEF.md` kill criteria).

### Task 11 — Web dashboard

**Files:** new `dashboard/` (FastAPI + whatever minimal frontend)

Three panels minimum: live agent-session feed (clean + hijacked), risk
verdict panel (flag + reason + contributing signals), metrics panel
(per-attack precision/recall, mutation-round curve). Wire to real output
from Tasks 5-10, not mocked data.

**Definition of done:** runs locally, shows real numbers from the actual
pipeline, not placeholder/hardcoded values.

### Task 12 — Integration contract + latency measurement

**Files:** new `defend/integration_contract.py`, doc: `INTEGRATION_CONTRACT.md`

Define the JSON payload the session-risk score would emit into a Decision
Intelligence-style consumer (per `TEAM_BRIEF.md` §9). Measure actual
end-to-end latency (all layers, single session) and log it.

**Definition of done:** a real measured latency number in milliseconds,
not an estimate, plus the JSON schema documented.

---

## 5. Testing conventions

- Every new module gets a corresponding test file (`tests/test_<module>.py`, `pytest`).
- Tests must run against small fixture data, not the full IEEE-CIS set (keep CI fast).
- No test should assert a specific AUC/F1 number as a hard pass/fail threshold except the explicitly named gates in §4 (Task 1, Task 5) — those thresholds are deliberate kill criteria, not arbitrary CI assertions.

## 6. Coding conventions

- Python 3.11+, type hints on all function signatures, dataclasses for structured data (matches `session_schema.py`'s existing style).
- No notebooks in the final repo — prototype in notebooks if useful, but port working code to `.py` modules before merging (a judge cross-checking the repo should find runnable scripts, not notebook cells).
- Every module-level docstring should state clearly what's proven vs. unverified, following the pattern already established in `divergence.py` — this repo has a habit of flagging its own confidence level, keep that habit going.
- Never fabricate a metric. If a number isn't computed, don't print a placeholder that looks computed.

## 7. Where this maps onto the calendar plan

If working against the dated schedule in `TEAM_BRIEF.md` Part 7: Tasks 1-3
≈ the Day 2 gate, Task 4 ≈ Days 3-4, Task 5 ≈ Day 5 hard gate, Task 6 ≈
started Day 3/finished Day 4 (parallel track), Task 7 ≈ Day 6-7, Task 8 ≈
Day 8-9, Task 9-10 ≈ Day 10-12, Task 11 ≈ ongoing from Day 7, Task 12 ≈
Day 9-10. Adjust to actual remaining days rather than treating these as
fixed dates — the task *order and dependencies* matter more than the exact
calendar mapping at this point.

## 8. Decisions not to make unilaterally — flag these back to the human instead

- Changing any field name/type in `AgentSession` or `MandateScope` (schema is locked; a change ripples through every downstream file).
- Substituting a different base dataset for IEEE-CIS.
- Deciding a kill-criterion gate "basically passed" when the measured number is below the stated threshold.
- Cutting one of the two attack types, or adding a third.
- Any change to what's claimed as "built" vs "documented only" in taxonomy/deck language — that distinction was deliberately fought for earlier and matters for submission honesty.
