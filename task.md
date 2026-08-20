# task.md

Running log of work done against `plan.md`'s task list. Update this after
finishing each task — what was actually done, the real measured result
(not a projected one), and what's next. `plan.md` / `TEAM_BRIEF.md` say
what *should* happen; this file records what *did* happen and when.

## Decision Log

Key decisions that deviate from or resolve an open question in
`plan.md`/`TEAM_BRIEF.md`, kept here as a single lookup point.

### 2026-08-20 — Primary signal pivoted from divergence to constraint_drift/trust_score

**Question asked:** "Given the divergence signal fails on subtle attacks
(AUC 0.348) while constraint_drift/trust_score hold up (0.99/0.97),
should `constraint_drift` + `ingestion_source_trust_score` be treated as
the primary detection signal going forward (updating README/task.md
language, and how downstream tasks like LightGBM feature selection get
framed)?"

**Answer: Yes — pivot now**, not waiting for the remaining ~80 sessions
(user's exact reply: "yes").

**Evidence it was based on** (n=130 real LLM-generated sessions, see the
2026-08-20 entries below for full detail): `utterance_artifact_divergence`
scores worse than random on subtle same-domain hijacks (subtle AUC=0.348)
while `constraint_drift` (0.988) and `ingestion_source_trust_score`
(0.967) both hold up on the same subtle subset.

**Confirmed at full sample size:** the remaining 80 sessions were
generated later the same day (via Ollama, see below) — at the full n=210,
divergence subtle AUC=0.419, constraint_drift=0.979,
ingestion_source_trust_score=0.905. Same conclusion, not a small-sample
artifact.

**What this changes going forward:** `constraint_drift` +
`ingestion_source_trust_score` (`defend/constraint_drift.py`) are the
primary Defend signal, not `utterance_artifact_divergence`
(`defend/divergence.py`) — per plan.md §8's own kill-criteria language.
This affects: which features get top billing in the eventual LightGBM
model (Task 5), and any deck/positioning language that currently calls
divergence the headline novelty claim (`TEAM_BRIEF.md` §4.3's framing of
`utterance_artifact_divergence` as "the core novelty claim" needs
revisiting against this result — flagged here, not silently changed in
that file since it's the positioning doc, not engineering state).
Divergence isn't deleted or abandoned — it still separates obvious
hijacks perfectly and remains a secondary signal — it's just no longer
the one the gate/headline claim leans on.

---

## 2026-08-19 — Repo state at init

No task work done yet this session. Current state per `plan.md` §2 /
`README.md`:

- `generate/session_schema.py` — done, locked.
- `generate/synthetic_sessions.py` — placeholder (5 legit + 5 hijacked,
  hand-written, not from the real narrative generator).
- `defend/divergence.py` — done, semantic path unverified in the sandbox
  that wrote it (no huggingface.co access there).
- `defend/isolation_test.py` — done. Last run: lexical-only AUC = 1.000,
  explicitly flagged as not trustworthy (semantic model didn't load,
  hand-written examples make separation artificially easy).
- `identify/` — empty.

**Next up: Task 1** (`plan.md` §4, "Confirm the real divergence signal").
Run `python3 defend/isolation_test.py` in an environment where the
semantic model actually loads, and read the SEMANTIC/COMBINED AUC lines.
This is the Day-2 gate — ≥0.85 → proceed to Task 2 (real narrative
generator); <0.85 → stop and jump to Task 3 (fallback signal), report
the failure plainly per the plan's kill criteria.

---

## 2026-08-19 — Task 1: real divergence signal confirmed

Environment: Python 3.14 (`python3` on PATH; bare `pip` is not — use
`python3 -m pip`). `numpy`/`pandas`/`scikit-learn`/`torch` were already
installed; installed the one missing dependency,
`python3 -m pip install -r requirements.txt` (pulled in
`sentence-transformers`). Confirmed the model loads:
`python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('OK')"`
→ `OK` (network to huggingface.co is reachable from this machine).

Ran `python3 defend/isolation_test.py` for real (semantic model loaded,
not the lexical-only fallback from before). Actual output:

```
LEXICAL (TF-IDF):       legit mean=0.668 std=0.076   hijack mean=1.000 std=0.000   AUC = 1.000
SEMANTIC (embeddings):  legit mean=0.194 std=0.053   hijack mean=0.673 std=0.044   AUC = 1.000
COMBINED (pipeline):    legit mean=0.336 std=0.044   hijack mean=0.771 std=0.031   AUC = 1.000
```

**Result: Semantic AUC = 1.000, Combined AUC = 1.000 — clears the ≥0.85
gate** with a real margin (legit 0.194 vs hijack 0.673 on the semantic
score alone, not just an artifact of TF-IDF word overlap). Per plan.md
Task 1's definition of done, this means proceed to Task 2.

Honest caveat, not glossed over: this is still the 10 hand-written
sessions from `generate/synthetic_sessions.py`, not the real
narrative-generator output. Plan.md is explicit that hand-written hijack
examples are typically easier to separate than realistic *subtle*
same-domain attacks — so this AUC=1.000 validates the mechanism works,
not that the production signal will hold at this strength. The real
Day-2-equivalent number is Task 2's obvious-vs-subtle split, still
pending. Task 2 itself (`generate/narrative_generator.py`) is blocked
in this environment — no `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` is set —
so it wasn't started; asking for one before proceeding.

Updated `README.md`'s "Current result" section to replace the stale
lexical-only note with these real numbers.

## 2026-08-19 — Task 3: fallback signal built (in parallel, per plan.md)

Per plan.md, Task 3 is built regardless of Task 1's outcome, not only as
an emergency fallback. Added `defend/constraint_drift.py`:

- `compute_constraint_drift(session)` — checks the INR amount embedded in
  `signed_artifact_text` against `mandate_scope.amount_cap`, and whether
  `task_origin_url`'s domain is in `mandate_scope.merchant_allowlist`.
  Returns a 0–1 score (violations / 2 checks), not just a bool.
- `compute_ingestion_source_trust_score(session)` — flags any
  `content_sources_ingested` domain that isn't an exact allowlist match;
  `typosquat_similarity()` exposes the underlying edit-distance ratio
  separately for inspection.

Note: `AgentSession`'s schema (locked) has no structured "actual amount /
actual merchant" field — only free-text `signed_artifact_text` — so the
amount check only fires when an INR figure is actually present in the
text; it does not (and structurally cannot, without violating the schema
lock) catch a violation encoded in a way the regex doesn't recognize.
This is a stated limitation of the heuristic, not a hidden gap.

Added `tests/test_constraint_drift.py` (new `tests/` dir, `pytest`, per
plan.md §5), using the existing `LEGITIMATE[0]`/`HIJACKED[0]` sessions
from `generate/synthetic_sessions.py` as the clear-pass/clear-violation
fixtures rather than inventing new ones. `python3 -m pytest
tests/test_constraint_drift.py -v` → **8/8 passed**. Added `pytest` to
`requirements.txt` since it's now a real dependency for running tests on
a clean clone.

**Next up:** Task 2 (real narrative generator) and Task 4 (IEEE-CIS join)
are both next in plan.md's build order but blocked on credentials not yet
configured here — need `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` for Task 2,
Kaggle API credentials (`~/.kaggle/kaggle.json`) for Task 4.

---

## 2026-08-20 — Kaggle + Gemini credentials configured

User provided a Kaggle API token (stored at `~/.kaggle/access_token`,
verified with `kaggle competitions list -s ieee-fraud-detection` — auth
works, competition confirmed reachable) and a Gemini API key (no
Anthropic/OpenAI key available, contrary to plan.md's original
assumption). Gemini key stored in a new repo-root `.env`
(`GOOGLE_API_KEY=...`, gitignored) and smoke-tested directly against
`gemini-2.5-flash` via the `google-genai` SDK — works.

Task 4 (IEEE-CIS download) intentionally not started yet — that's a
~600MB+ download and needs its own explicit go-ahead given the size.

## 2026-08-20 — Task 2: adapter + generator built, blocked mid-run by Gemini free-tier daily quota

Per user's request, the LLM call is behind an adapter
(`generate/llm_adapter.py`: `LLMAdapter` ABC, `GeminiAdapter`
implementation, `get_default_adapter()` factory) so the provider can be
swapped later without touching generation logic. Added
`injection_payload_text: Optional[str] = None` to `AgentSession` — the one
confirmed, deliberate exception to the schema lock (purely additive,
confirmed with user before making the change).

Built `generate/narrative_generator.py` (`generate_benign_session`,
`generate_hijacked_session` with `subtlety="obvious"|"subtle"`, 10 domains
per plan.md) and `generate/generated_sessions.py` (orchestrates + caches
to `data/generated_sessions.json`, saves incrementally so a mid-run
failure doesn't lose progress, resumes from a partial cache on rerun).

**Hit a real, hard blocker partway through the pilot run (target: 30
sessions, `n_per_slot=1`):** this Gemini API key is on the free tier,
capped at **20 requests/day per model** (not a per-minute limit — a
`GenerateRequestsPerDayPerProjectPerModel-FreeTier` quota). Burned through
`gemini-2.5-flash`'s 20/day during initial smoke-testing, switched the
adapter's default model to `gemini-2.5-flash-lite` (separate quota
bucket), and hit that model's own 20/day cap after 17/30 pilot sessions.
Both models are now exhausted for today on this key.

**The 17 real sessions generated before hitting quota are still a
genuine, useful signal** (all real LLM output, not hand-written) — ran
`defend/isolation_test_generated.py`-equivalent scoring on them:

```
OBVIOUS subset vs legit: n=12  AUC = 1.000
SUBTLE  subset vs legit: n=11  AUC = 0.733
```

This is exactly the risk plan.md flagged: the subtle case (same domain,
only amount/merchant/destination shifted) separates meaningfully worse
than the obvious case, and **0.733 is below the 0.85 gate**. Sample size
is small (n=11 for subtle) so this isn't the final Task 2 number — plan.md
wants ~200 sessions — but it's a real early warning, not something to
paper over. If this pattern holds at full sample size, per plan.md §8 /
TEAM_BRIEF.md Part 8 the correct move is pivoting `constraint_drift` +
`ingestion_source_trust_score` to the primary signal, not forcing
divergence to work.

**Blocked on a decision only the user can make:** how to get from 17 to
~200 real sessions given the 20/day/model free-tier cap — wait for daily
resets (slow, ~10+ days for one model), enable billing on the Gemini key
(fast, their account/cost decision), or accept a smaller sample size.
Paused here pending that decision rather than guessing.

---

## 2026-08-20 — Switched to Groq; hit the same wall class, then a real result at n=127

User opted to use Groq instead of continuing to fight Gemini's quota
(they have a Groq key, no Anthropic/OpenAI key). Added `GroqAdapter` to
`generate/llm_adapter.py` (same `LLMAdapter` interface — this is exactly
what the adapter pattern was for). Groq uses OpenAI-compatible strict
JSON-schema mode; had to inject `additionalProperties: false` into every
object node adapter-side (`_require_no_additional_properties`) so
`narrative_generator.py`'s schemas stay plain/provider-agnostic rather
than carrying a Groq-specific quirk.

**Same problem, different provider:** `openai/gpt-oss-120b` (initial
default model) hit Groq's free-tier **200,000-tokens/day** cap during
testing. Switched default model to `openai/gpt-oss-20b` (separate quota
bucket) — that got a clean 30-session pilot, then 43/210 into the full
run before **also** hitting its own 200k TPD cap. Checked the other
models on this key: `qwen/qwen3.6-27b` reliably fails strict JSON-schema
validation (reasoning-model output doesn't validate), `allam-2-7b`
doesn't support `json_schema` response format at all. So both viable
Groq models are now exhausted for today too.

Along the way, fixed a real bug this surfaced: an early "stuck" 35-minute
background run (task `bqj3pj6c3`) wasn't actually hung — `GroqAdapter`
was silently sleeping through multi-minute rate-limit retries with no
visible progress, indistinguishable from a hang. Fixed by making
`generate_json` fail fast (`GroqQuotaExhausted`) instead of sleeping past
20s on a suggested retry delay — a real quota exhaustion should surface
immediately, not look like a freeze. Also fixed the incremental-cache
+ resume logic in `generate/generated_sessions.py` (was previously
all-or-nothing) so each of these partial failures kept its progress
instead of losing it, and a rerun resumes rather than restarts —
without this, we'd have lost 43 sessions and then 127 sessions to the
two quota walls instead of keeping them.

**Net result before the second wall: 127/210 real sessions generated**
(43 legit, 42 obvious-hijack, 42 subtle-hijack — well balanced). Running
the same evaluation as before on this larger, real sample:

```
utterance_artifact_divergence:
  OVERALL: n=127  legit_mean=0.375  hijack_mean=0.556  AUC=0.674
  OBVIOUS: n=85   legit_mean=0.375  hijack_mean=0.776  AUC=1.000
  SUBTLE:  n=85   legit_mean=0.375  hijack_mean=0.336  AUC=0.348
```

**This is the real Day-2-equivalent finding, and it's stark: on subtle
attacks, divergence is worse than random (0.348 < 0.5)** — the subtle
hijacked sessions' mean divergence score is actually *lower* than the
legit mean. Same-domain, close-wording hijacks (the realistic attack
plan.md flagged as the one that matters) read as *more* semantically
similar to the original utterance than ordinary benign paraphrasing does,
which is the opposite of what the signal needs. n=85 per group is not
tiny — this isn't sample-size noise, it's a real mechanism failure.

**Then evaluated the Task 3 fallback signal on the exact same 127
sessions** (zero extra API cost — pure local computation) to see whether
it holds up where divergence didn't. First hit a real bug doing this:
`extract_amount_inr` crashed on real LLM output phrased as "250000 INR"
(number before the currency code — the regex only handled "INR 3000"
order) and, worse, `"INR,"` with no digit following matched the comma
alone and crashed on `float('')`. Fixed in `defend/constraint_drift.py`:
regex now matches either order, requires a leading digit, and takes the
*max* amount mentioned (a session can state several figures). Added 3
regression tests (`tests/test_constraint_drift.py`, now 11/11 passing)
covering exactly this failure mode so it can't silently regress.

With that fixed:

```
constraint_drift:
  OVERALL: n=127  legit_mean=0.000  hijack_mean=0.869  AUC=0.994
  OBVIOUS: n=85   legit_mean=0.000  hijack_mean=0.976  AUC=1.000
  SUBTLE:  n=85   legit_mean=0.000  hijack_mean=0.762  AUC=0.988

ingestion_source_trust_score:
  OVERALL: n=127  legit_mean=0.058  hijack_mean=0.952  AUC=0.979
  OBVIOUS: n=85   legit_mean=0.058  hijack_mean=1.000  AUC=0.988
  SUBTLE:  n=85   legit_mean=0.058  hijack_mean=0.905  AUC=0.969
```

**Both fallback signals hold up on subtle attacks where divergence
collapses** (0.988 and 0.969 vs 0.348). This makes intuitive sense: a
subtle hijack keeps the wording semantically close by design (that's
what makes it subtle to divergence), but it still has to blow past the
mandate's amount cap or move off the merchant allowlist to actually steal
money — which is exactly what `constraint_drift` checks directly, no
embedding model needed.

**Per plan.md §8 / TEAM_BRIEF.md Part 8's own kill criteria, this is the
signal to act on.** Not declaring this decided unilaterally — pivoting
which signal is "primary" for the project is flagged in plan.md as a
call for the human, not something to silently change in code — but the
evidence at n=127 is strong enough that more data is unlikely to reverse
the qualitative conclusion.

**Still open:** 83 of the planned ~200 sessions remain ungenerated
(blocked on the same daily-quota class of issue, now on a second
provider). Paused generation pending user direction — options are
waiting for tomorrow's quota reset, upgrading a provider's tier, or
treating n=127 as sufficient for this finding and moving on.

## 2026-08-20 — Fixed isolation_test_generated.py triggering generation as a side effect

Running `defend/isolation_test_generated.py` to check the result called
`generate_dataset()` internally, which — since the cache had 127 < the
default target of 210 — tried to generate the missing 83 sessions itself,
immediately re-hitting the fully exhausted Groq quota and crashing (after
squeezing in 3 more first: 127 → 130). An evaluation script shouldn't
have the side effect of spending API calls just by being run. Added
`generate.generated_sessions.load_cached_dataset()` (loads whatever's
cached, never generates) and switched `isolation_test_generated.py` to
use it. Also extended the script to report all three signals side by
side (`utterance_artifact_divergence`, `constraint_drift`,
`ingestion_source_trust_score`), since that three-way comparison is the
actual point, not divergence in isolation.

**Final numbers for this session, n=130** (46 legit / 42 obvious / 42
subtle — essentially unchanged from the n=127 read, confirming it wasn't
a fluke of the smaller sample):

| Signal | Overall AUC | Obvious AUC | Subtle AUC |
|---|---|---|---|
| `utterance_artifact_divergence` | 0.674 | 1.000 | **0.348** |
| `constraint_drift` | 0.994 | 1.000 | 0.988 |
| `ingestion_source_trust_score` | 0.978 | 0.989 | 0.967 |

To resume generation later: `python3 -m generate.generated_sessions 7`
(picks up from the 130 cached, no `--force`). To re-check results without
spending any API calls: `python3 defend/isolation_test_generated.py`.

## 2026-08-20 — Added OllamaAdapter; Task 2 complete at full n=210

User's third pick for the LLM provider: local inference via Ollama
(already installed, RTX 3050 6GB VRAM), no daily quota since it runs on
this machine rather than a hosted API. Added `OllamaAdapter` to
`generate/llm_adapter.py` — same `LLMAdapter` interface, third
implementation, no changes needed anywhere else (this is now three
providers deep and the adapter pattern has paid for itself each time).

The only model already pulled (`minicpm-v`, a vision model) produced
unusable output (literal `"None"` strings in required fields) — not
suited to this task. Checked available VRAM before picking a replacement:
6GB rules out a comfortable 7-8B model (quantized ~4.7GB leaves little
headroom, risks slow CPU offload), so pulled `qwen2.5:3b-instruct`
(~1.9GB) instead. That model needed one adapter-side fix: with a plain
user-only prompt it drifted into replying as a customer-service chatbot
(a literal greeting) instead of filling out session fields; adding a
fixed system message ("output ONLY a JSON record... not a chatbot...")
fixed this completely and also cut latency (23.6s → ~5-8s/call, first
call included model warm-up). This was a model-specific quirk, so the
fix lives in `OllamaAdapter`, not in `narrative_generator.py`'s prompts.

Set `USE_OLLAMA=1` in `.env` (checked first in `get_default_adapter()`),
resumed generation from the 130 cached sessions, and it completed the
full 210-session dataset (70 legit / 70 obvious / 70 subtle, exactly
balanced) with no quota issues at all — confirms local inference is the
right call for large generation runs going forward, hosted free tiers
are fine for the earlier day's spot-checks.

**Final result, n=210 (up from the interim n=130 read — confirms it
wasn't a small-sample artifact):**

| Signal | Overall AUC | Obvious AUC | Subtle AUC |
|---|---|---|---|
| `utterance_artifact_divergence` | 0.710 | 1.000 | **0.419** |
| `constraint_drift` | 0.989 | 1.000 | 0.979 |
| `ingestion_source_trust_score` | 0.942 | 0.979 | 0.905 |

Same conclusion as the interim read, now at the plan's target sample
size: divergence separates obvious hijacks perfectly but stays well
below a useful threshold on subtle same-domain attacks (0.419, barely
above chance); both `constraint_drift` and `ingestion_source_trust_score`
hold up strongly on subtle attacks too (0.979 / 0.905). This is now the
final Task 2 deliverable — the real Day-2-equivalent result plan.md asked
for, at the real target sample size, confirming the 2026-08-20 pivot
decision logged above rather than just informing it.

**Task 2 is complete.** Reproduce anytime with
`python3 defend/isolation_test_generated.py` (reads the cache, no API
calls). Next up per `plan.md`: Task 4 (IEEE-CIS join — Kaggle credentials
already configured, download not yet run) or formalizing the signal
pivot in `TEAM_BRIEF.md`-facing language (flagged in the Decision Log
above as not yet done).

## 2026-08-20 — Task 4 started: IEEE-CIS download (blocked, then in progress); TEAM_BRIEF.md citations verified and extended

Kaggle CLI download of `train_transaction.csv`/`train_identity.csv`
(only the labeled files — test files have no `isFraud` label and aren't
needed since we're not submitting to Kaggle's leaderboard) initially
failed with `403 Forbidden` even though auth worked
(`kaggle competitions list` succeeded) — Kaggle requires accepting a
competition's rules on the website before the API serves its data,
separately from account authentication. Declined a suggested workaround
(connecting a Kaggle-hosted remote MCP server) since it wouldn't bypass
an account-level rules-acceptance gate regardless of client, and this
session can't complete a new server's OAuth flow anyway. User accepted
the rules on kaggle.com directly and is downloading the competition zip
manually; the earlier researched alternative sources (Hugging Face has no
mirror, IEEE DataPort requires a paid subscription and just points back
to Kaggle) confirmed Kaggle is the only real option.

Separately, verified every citation in `TEAM_BRIEF.md` §1.3/§1.4/Part 10
actually exists (arXiv 2601.22569, CSA's Oct-2025 AP2 report, Halborn's
two AP2 posts, the Ravelin n=1,504 and Darwinium n=500 surveys — all
real, numbers match). Found and added two missing citations at the
user's request:
- **Greshake et al. (arXiv 2302.12173, 2023)** — the foundational paper
  naming indirect prompt injection as an attack class — added to §1.3
  and to attack #47's grounding in Part 10.
- A proper academic reference for attack #48 (arXiv 2307.05633,
  "Transaction Fraud Detection via an Adaptive Graph Neural Network"),
  added alongside the existing "top Kaggle solutions" grounding — noted
  in the doc itself that #48's grounding is still thinner than #47's, no
  paper covers this exact card-rail merchant-network setup directly.

## 2026-08-20 — Task 4 complete: IEEE-CIS join + Rules layer

User accepted the Kaggle competition rules on their account (the 403 was
purely that account-level gate, unrelated to the API token itself) and
downloaded `train_transaction.csv` (652MB, 590,540 rows) +
`train_identity.csv` (26MB) into `data/ieee-fraud-detection/`. Verified
the schema matches real IEEE-CIS before building anything on top of it
(`isFraud`, `TransactionAmt`, `card1-6`, `C1-14`, `D1-15`, `M1-9`,
`V1-339`; identity file has `id_01-38`, `DeviceType`, `DeviceInfo`).

Built `generate/join_ieee_cis.py`: left-joins transaction+identity on
`TransactionID`, samples 210 rows **only from `isFraud == 0`** real
transactions (fixed seed, reproducible) to pair with the 210 generated
sessions — deliberately not sampling from IEEE-CIS's own fraud-flagged
rows, since the project's thesis is that a hijacked agent payment looks
like an *ordinary* transaction on the card rail; pairing with a
classic-fraud row would muddy that and hand Task 5's LightGBM a spurious
shortcut feature. Scores each session (divergence + both Task 3 signals)
before calling the locked `to_row()`, so the joined output carries real
computed-signal values, not `None`. Output: `data/joined_sessions.csv`
(210 rows × 448 columns).

**Real bug caught while building this:** wrote the benign sessions'
subtlety tag as the generator's internal `"n/a"` sentinel straight into
the CSV — pandas' default `read_csv` NA-value list includes `"n/a"`, so
every benign row's subtlety silently became `NaN` on a normal load
(confirmed by actually loading the output and checking, not assumed).
Fixed with `_relabel_benign_subtlety()` (`"n/a"` → `"benign"` at the CSV
boundary only — the JSON cache and in-memory code elsewhere are
untouched), with a regression test
(`tests/test_join_ieee_cis.py::test_relabel_benign_subtlety_avoids_pandas_na_string`)
that actually round-trips through `pd.read_csv` rather than just
asserting the string value.

Built `defend/rules.py` — Defend layer 1, hard checks, no ML: amount over
`mandate_scope.amount_cap`, `task_origin_url` domain not in
`mandate_scope.merchant_allowlist`, `agent_registry_status != "valid"`.
Reuses `constraint_drift.py`'s `extract_amount_inr`/`extract_domain`
rather than reimplementing them. No separate "category violation" check —
in this dataset it would be redundant with the merchant-allowlist check
(each domain maps 1:1 to its own allowlist), said so in the docstring
rather than adding a rule that doesn't add independent signal.

**Result on the full 210-session set — comfortably clears the Task 4
definition of done** ("recall on hijacked-obvious meaningfully non-zero"):

| Subset | Flagged | Rate |
|---|---|---|
| benign (n=70) | 0/70 | 0% (zero false positives) |
| obvious-hijack (n=70) | 70/70 | 100% recall |
| subtle-hijack (n=70) | 67/70 | 95.7% recall |

Even this simple, explainable, no-ML layer catches nearly all subtle
hijacks too — consistent with Task 3's constraint_drift AUC finding
(0.979 on subtle), since both are checking the same underlying mandate
violations. Added `tests/test_rules.py` (uses the existing hand-written
`LEGITIMATE`/`HIJACKED` fixtures, no new ones needed) and
`tests/test_join_ieee_cis.py` (tiny in-memory fixture, not the real
650MB file, per `plan.md` §5's "keep CI fast" rule) — full suite now
19/19 passing.

**Task 4 is complete.** Next per `plan.md`: Task 5, the LightGBM baseline
(the hard gate) — consumes `data/joined_sessions.csv` directly, with
`constraint_drift`/`ingestion_source_trust_score` as the primary features
per the confirmed pivot.

## 2026-08-20 — Task 5 complete: LightGBM baseline (the hard gate) — cleared, after catching a real label-leakage bug

Installed `lightgbm` (native win_amd64 wheel, no compatibility issue).
Built `defend/lightgbm_baseline.py` (`build_feature_matrix` — parses the
CSV-string-repr list columns from Task 4's join, engineers
`mandate_category`/`mandate_allowlist_size`/`task_origin_domain`/
`num_content_sources`, casts high-cardinality IEEE-CIS columns to
categorical for LightGBM's native handling; `train_baseline` — one
full-data fit for feature importance only) and `defend/evaluation.py`
(stratified 5-fold CV on the 3-way benign/obvious/subtle label,
aggregated out-of-fold predictions — chosen over a single 80/20 split
because at n=210 a single split leaves too few test rows per class to
trust; every row here gets scored exactly once by a model that never
trained on it).

**First run produced a suspicious AUC=1.000 across every subset.**
Feature importance immediately explained why: `hops_since_intent` alone
accounted for effectively 100% of total gain, with `constraint_drift`
(the actual primary signal) showing *zero* importance. Traced it to
`narrative_generator.py`'s `_base_session()`
(`hops_since_intent=random.randint(1, 4) if injection_present else 0`) —
I had written this in Task 2 without realizing it makes the field a
deterministic encoding of the label itself (0 iff benign), not a
narrative-derived signal. This is exactly the kind of fabricated-looking
metric plan.md's rules exist to catch, and it would have been very easy
to just report "AUC=1.000, LightGBM gate cleared" without ever looking at
feature importance — reporting feature importance wasn't optional
overhead, it's what caught this.

**Fixed properly, not patched around:**
- `narrative_generator.py`: `hops_since_intent` now draws from
  `random.randint(0, 4)` unconditionally, same distribution regardless of
  `injection_present`.
- Patched the existing 210-session cache's `hops_since_intent` values
  directly (pure Python bookkeeping, not LLM-generated content — no need
  to re-spend any API/local-inference calls to fix this).
- Rebuilt `data/joined_sessions.csv` and reran evaluation.
- Added a regression test
  (`tests/test_narrative_generator.py::test_hops_since_intent_not_conditioned_on_injection_present`)
  that directly checks `_base_session` can produce `hops_since_intent==0`
  even when `injection_present=True`, so this exact pattern can't come
  back silently.

**Real result after the fix (stratified 5-fold CV, out-of-fold, n=210,
443 features):**

| Subset | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Overall | 0.993 | 0.957 | 0.975 | 0.986 |
| Obvious vs benign | 0.986 | 1.000 | 0.993 | 1.000 |
| Subtle vs benign | 0.985 | 0.914 | 0.948 | 0.971 |

**False-positive rate on legitimate traffic: 0.014 (1/70)** —
`TEAM_BRIEF.md` §4.8, reported as its own line, not folded into overall
accuracy.

**Feature importance (gain, full-data fit) now makes sense and directly
validates the project's thesis:** `constraint_drift` dominates (1276.8,
~93% of total gain), `ingestion_source_trust_score` (22.2) and
`utterance_artifact_divergence` (6.3) contribute meaningfully; the real
IEEE-CIS transaction columns (`TransactionDT`, `M8`, `D4`, `D2`, `D10`,
`card1`, `V61`, ...) show small but non-zero importance. This is the
concrete evidence for `TEAM_BRIEF.md` §1.2's central claim: classic
transaction-level features barely help here, the novel session-level
signals do almost all the work.

Small-sample caveat stated plainly (n=210, 443 features — a real
small-n-large-p situation): this is a first real read of feasibility, not
a statistically bulletproof production number.

**Task 5 is complete — the hard gate is cleared with real, honest
numbers, including the leakage bug that got caught rather than missed.**
Next per `plan.md`: Task 6 (Identify RAG agent) is independent/parallel;
Task 7 (content/semantic layer) and Task 8 (GNN, attack #2 — no data yet)
are further out.

## 2026-08-20 — Task 6 complete: Identify RAG agent (LangGraph + Groq/Ollama + Qdrant + networkx)

Per user direction: built fresh rather than importing `plan.md`'s
referenced `production_rag` pipeline. Found the real candidate at
`D:\rag` (FastAPI + LangChain + ChromaDB + Groq + sentence-transformers
per its own README) — but most files are empty scaffolds and it never
actually implemented the LangGraph orchestration its README describes as
the architecture. Mirrored its separation-of-concerns pattern (loader →
chunker → embedder → vector store) in fresh code, with LangGraph actually
implemented this time, Qdrant instead of Chroma (local/on-disk, no
server), and Groq/Ollama via our own already-working
`generate/llm_adapter.py` (reused as-is — third provider added to that
same interface with zero changes elsewhere, which is the whole point of
having built it that way in Task 2).

**Decision confirmed with user first:** no 46/48-item taxonomy markdown
exists anywhere on this machine (searched); `discover.py` reports
however many genuine, well-grounded entries the real corpus yields, not
padded to 48.

**Knowledge graph addition:** initially recommended against a knowledge
graph for this task (embedding similarity seemed sufficient at this
scale). User pushed back and asked to check the research properly —
correctly. Found real published precedent (CyKG-RAG, AgCyRAG, MITRE
ATT&CK-to-CVE knowledge graphs) showing GraphRAG-style hybrid graph+vector
approaches measurably beat vector-only RAG specifically on
cross-document aggregation/consolidation tasks (3-4x better on
aggregation queries per multiple 2026 benchmarks) — which is exactly
what "read many threat reports, consolidate overlapping attacks" is.
Added `identify/knowledge_graph.py` (in-memory `networkx.MultiDiGraph`,
no Neo4j server — same "no extra infra" principle as Qdrant's local
mode), used for hybrid novelty scoring alongside embedding similarity.

**Corpus: 9 real, individually verified documents** (not the aspirational
30-40 — fetched via WebFetch, saved with source URLs in
`identify/corpus/`): Greshake et al. (arXiv 2302.12173, the foundational
indirect-prompt-injection paper), arXiv 2601.22569 ("Whispers of
Wealth"), CSA's AP2 STRIDE/MAESTRO report, two Halborn posts, Ravelin's
2026 survey, Darwinium's 2026 report, Visa's Spring 2026 Threats Report,
Zscaler ThreatLabz's indirect-prompt-injection post. Mastercard's
Verifiable Intent page returned HTTP 403 / no fetchable content on two
attempts — skipped rather than fabricated. Plus a sample (2 per
domain×subtlety combination, ~40 documents) of real `injection_payload_text`
values from Task 2's `generated_sessions.json`, per `plan.md`'s explicit
instruction to include those.

**Operational constraint:** Groq's `gpt-oss-120b`/`gpt-oss-20b` were both
still at their 200k-token daily cap from Task 2. `discover.py` defaults
to `OllamaAdapter` (`qwen2.5:3b-instruct`) for extraction — same
`LLMAdapter` interface, so switching back to Groq once quota resets is a
one-line change.

**Real bug caught during this task, fixed with a regression test:** the
first post-hoc consolidation pass (`consolidate_by_name`, added because
the small local model doesn't tag `delivery_vector`/`mechanism_type`
consistently across repeated mentions of the same real attack, so the
primary hybrid novelty scorer under-caught duplicates) used name
similarity alone and **incorrectly merged "Branded Whisper Attack" with
"Vault Whisper Attack"** — two genuinely different real attacks from the
same paper (one manipulates product rankings, the other exfiltrates
data) that happen to share a "Whisper Attack" name suffix (0.76 string
similarity). Caught by manually inspecting the consolidated output
against the source documents rather than trusting the count. Fixed by
requiring BOTH name similarity AND mechanism-text similarity above a
floor before merging; added
`tests/test_consolidate_by_name.py::test_does_not_merge_similar_names_different_mechanisms`
as a regression test naming this exact case. Reran the full pipeline
after the fix — 30/30 tests passing throughout.

**Final real result:**
- Raw discovery (before consolidation): 88 entries.
- After consolidation: **61 attack entries** in `identify/taxonomy.json`.
- **32 grounded in at least one real threat-report URL**; 29 grounded
  only in our own synthetic injection-payload samples (still real,
  LLM-generated content from Task 2, just not independently corroborated
  by external threat intel).
- Knowledge graph: 207 nodes, 281 edges, exported to
  `identify/knowledge_graph.json` (the raw discovery trace — not
  consolidated, unlike `taxonomy.json`; kept as an inspectable record of
  the actual discovery process).

**Known quality limitations, stated plainly rather than glossed over:**
a 3B-parameter local model is noticeably less consistent than the
Groq-hosted models used earlier — category-tag formatting varies
(`"merchant impersonation"` vs `"merchant-impersonation"` as separate
strings), some entry names are awkward (`"Indirect Whisper Whisper
Attack"`), and extraction isn't fully reproducible run-to-run (a rerun
after the consolidation fix didn't re-surface "Vault Whisper Attack" as
a separate entry at all, likely absorbed into a differently-phrased
extraction that round). This is a real, disclosed limitation of using
local inference for extraction quality, not a hidden one — the mechanism
(LangGraph loop, real corpus, real grounding, hybrid dedup) is sound and
would produce cleaner output with a larger model once Groq's quota
allows it.

**Task 6 is complete.** Reproduce with `python3 -m identify.discover`
(takes ~15-17 min via local Ollama, 67 chunks × ~1-2 LLM calls each). Test
suite: `python3 -m pytest tests/ -v` → 30/30 passing.

## 2026-08-20 — Task 7 complete: content/semantic injection-detection layer

Per `plan.md`'s explicit instruction, searched for the referenced
"notification-router Jaccard-similarity" prior work before building
anything — found a real, substantial project at
`D:\orchestrate_learning\a\Message-Notification-Router`. Read (not
imported) the actual algorithm: `router/evidence.py`'s `_similarity()` —
`len(a & b) / len(a | b)` on lowercased regex-tokenized word sets — and
the confidence-calibration methodology in `eval/metrics.py`
(`brier_score()`, `expected_calibration_error()`), which
`EVALUATION.md` reports as real numbers (Brier 0.0925, ECE 0.0500), not
an abstract idea.

Per `TEAM_BRIEF.md` §4.6, this layer is specifically about "injection
payloads in ingested page content" — distinct from Task 3's domain/
typosquat checks. Real constraint worth stating plainly:
`AgentSession` has no actual scraped-page-text field, only
`content_sources_ingested` (domain strings). The one real text field
standing in for "injected content" is `injection_payload_text` (hijacked
sessions only, `None` for benign). So `defend/content_layer.py`'s actual
task is: does a piece of text read like a covert instruction-override
vs. ordinary session content — scored via `injection_payload_text`
(hijacked) vs `raw_utterance` (benign), stated as the proxy it is.

`REFERENCE_INJECTION_PHRASES` in `content_layer.py`: 4 real quotes from
Task 6's fetched corpus (Zscaler's quoted hidden-CSS instruction text,
Halborn's "bury or override mandate constraints," CSA's "compromised
instructions redirect agent delegation") — not fabricated placeholders.
`content_layer_eval.py` extends this with 20 real held-out
`injection_payload_text` values (2 per domain, sorted by `agent_id` for
determinism) to build a fuller reference set, then scores the
*remaining* 120 hijacked sessions + all 70 benign sessions — no session
appears in both the reference-building and eval sets.

**Real result:**

```
Threshold=0.15 (empirically swept 0.05-0.30): precision=0.983 recall=0.992 F1=0.988  (TP=119 FP=2 FN=1)
Confidence calibration: Brier score = 0.2485   ECE = 0.3705
```

**An honest, worthwhile finding, not smoothed over:** the raw Jaccard
score is an excellent *discriminator* (threshold it at 0.15 and
precision/recall are both ~0.99) but a poor *calibrated confidence* —
Brier 0.2485 is barely better than the 0.25 "uninformative" baseline,
and ECE 0.37 is large. Verified the Brier/ECE implementations against
known sanity cases (always-guess-0.5 → Brier=0.25 exactly;
perfectly-calibrated inputs → ECE=0) before trusting the number, rather
than assuming a bug when the calibration result looked bad. This
actually explains *why* the reference project uses fixed per-tier
confidence constants instead of exposing raw scores directly (its own
`reliability_curve()` exists specifically to check whether those
constants are calibrated) — the same lesson applies here: this score is
safe to threshold for a flag, not safe to hand to a risk analyst as "the
model is 35% confident," without a proper calibration mapping first
(not built — out of scope for Task 7's definition of done, which asked
for precision/recall like the other layers, not a calibrated production
confidence).

Added `tests/test_content_layer.py` (4 tests, comparative rather than
fixed-threshold since the unit-test default reference set is smaller
than the eval script's extended one). Full suite: 34/34 passing.

**Task 7 is complete.** Reproduce with `python3 -m defend.content_layer_eval`.
Next per `plan.md`: Task 8 (GNN, attack #2 — no generated data yet, needs
`generate/merchant_network.py` first) or Task 9/10 (LLM verdict / Mutator,
both further downstream).

## 2026-08-20 — Task 8 complete: GNN over merchant/acquirer/beneficiary graph (attack #2)

First generated data for attack #2 (transaction laundering via
fraudulent merchant network) — everything through Task 7 was attack #1
only. Per `TEAM_BRIEF.md` §2.1's stated connection between the two
attacks ("the typosquat storefront the hijacked agent buys from IS a
node in this network"), grounded the merchant nodes in real data rather
than inventing names: extracted **212 unique real malicious/typosquat
domains** from the 210 already-generated sessions'
`task_origin_url`/`content_sources_ingested`, plus the **19 real
legitimate domains** from `narrative_generator.py`'s `DOMAINS`.

**Real bug caught immediately, before trusting the node counts:** the
first extraction pass showed only 6 legitimate merchants instead of 19
— hijacked sessions sometimes reference the genuine origin domain
*alongside* the fake one in `content_sources_ingested` (e.g. the agent
"visited both"), so 13 real merchants were getting swept into the
"malicious" set and had their graph-node attributes overwritten when the
ring-building loop ran second. Fixed by excluding known-legitimate
domains from the malicious-domain extraction; added
`tests/test_merchant_network.py::test_legitimate_domains_not_counted_as_malicious`
as a regression test that checks the actual set overlap, not just node
counts.

`generate/merchant_network.py` builds the synthetic shell-merchant
network around those real domains: `Merchant --uses_acquirer--> Acquirer`,
`Merchant --owned_by--> Beneficiary`, 199 malicious domains partitioned
into rings (shared beneficiary + rapid/clustered registration, 1-60 days
ago) vs. 19 legitimate merchants (established, 500-3000 days ago).

**A second honesty catch, this time about the graph design itself, not a
bug:** the first version gave every legitimate merchant a fully unique
Acquirer, which made "does this merchant's acquirer have >1 connection"
alone a perfect classifier — the GNN hit AUC=1.000, but that validated
nothing beyond "an intentionally trivial synthetic graph is trivial,"
the same lesson Task 2's hand-written-vs-generated hijack examples
already taught this project. Fixed for realism: legitimate merchants now
draw from a small shared pool of 7 Acquirers (real small businesses
genuinely do share popular payment processors) but never share a
Beneficiary; larger rings (>6 members) split across 2 Acquirers
(spreading risk) while keeping one shared Beneficiary — shared
beneficial ownership carries more of the real discriminating power now,
matching `TEAM_BRIEF.md` Part 10 item #48's framing of shared
acquirer *and* beneficiary *and* rapid registration together, not any
one signal alone.

`defend/gnn.py`: `torch_geometric` (2.8.0.post1) installed cleanly — a
pure-Python wheel, no C++ extension compile, much less fragile than
PyG's reputation suggested. Built as a **homogeneous** graph (one-hot
node-type + registration-age features) with a 2-layer `GraphSAGE`,
deliberately not a strictly-typed `HeteroData`/`HeteroConv` — message
passing still runs over the real merchant/acquirer/beneficiary edges,
just without per-edge-type convolutions, chosen for reliability given
`plan.md`'s own instability-is-expected framing for this task. Training
converged cleanly (loss 0.674 → 0.022 over 100 epochs) — **the GNN path
was used, the LightGBM-on-graph-features fallback was not needed.**

**Real result** (transductive node classification, 70/30 stratified
split on node labels, held-out n=66):

```
precision=0.968  recall=1.000  F1=0.984  AUC=1.000
```

**Honest caveat on this number, stated plainly rather than left to
imply more than it should:** shared-beneficiary is still a strong,
close-to-deterministic signal in this graph by construction (no legit
merchant ever coincidentally shares one; every ring member always does)
— which is actually defensible as a modeling choice (shared beneficial
ownership genuinely is the canonical "follow the money" signal in real
AML/KYC investigation, and `TEAM_BRIEF.md` itself names it as one of the
three defining ring characteristics), but it does mean this result
reflects a ring that hasn't tried to obscure ownership. A more
adversarial ring (nominee directors, deliberately distinct-looking
ownership records) would be a harder, more realistic case — exactly the
kind of "harder round" the eventual Mutator (Task 10) is meant to
produce, not something manufactured here just to make the number lower.

Added `tests/test_merchant_network.py` (5 tests: the domain-overlap
regression, ring-sharing structure, legit-acquirer-pool realism,
registration-age clustering) and `tests/test_gnn.py` (2 tests: PyG data
shape consistency, training converges with real metrics returned). Full
suite: **41/41 passing**.

**Task 8 is complete — the GNN path succeeded, no fallback needed.**
Reproduce with `python3 -m generate.merchant_network` then
`python3 -m defend.gnn`. Next per `plan.md`: Task 9 (LLM verdict layer,
optional/has a SHAP fallback) or Task 10 (Mutator — the closed-loop
piece, feeds Defend's misses back into Generate/Identify).

## 2026-08-20 — Task 9 complete: LLM verdict layer shipped (SHAP fallback not needed)

Fused all four already-built Defend layers into one plain-English
verdict, reusing each as-is rather than reimplementing anything:
`defend/rules.py`'s `apply_rules()` (layer 1), a fresh
`cross_validated_predictions()` run from `defend/evaluation.py` (layer
2), `defend/content_layer.py`'s `score_injection_likelihood()` (layer
3), and a new `defend/gnn.py::predict_all_merchants()` (layer 4 — trains
on all node labels, not held out, since this is for operational scoring
rather than the Task 8 accuracy claim; kept as a separate function so it
can never be mistaken for `train_and_evaluate()`'s held-out result).

Set a concrete acceptance bar before deciding LLM-vs-fallback, per
`plan.md`'s "never ship a half-working layer, pick the fallback
explicitly" instruction: run on 12 representative sessions (4 benign, 4
obvious-hijack, 4 subtle-hijack) and check the stated `risk_level` is
directionally consistent with the four numeric signals
(`consistency_check()` in `defend/llm_verdict.py`), not just eyeballed.

**Result: 12/12 passed the consistency bar on both attempts (before and
after a prompt fix below) — the LLM verdict layer shipped, the
`defend/shap_fallback.py` fallback was not needed.** Used
`OllamaAdapter` (Groq's `gpt-oss-120b`/`gpt-oss-20b` are still at their
daily cap from Task 2 — confirmed by checking which adapter class
`_get_adapter_with_fallback()` actually returned, not assumed). Sample
explanation, correctly grounded in the real evidence given, not
invented: *"This transaction deviates from the user's request for a
prescription and instead orders a high-priced weight loss supplement.
The amount exceeds the 2000 rupee cap, and the domain
'medifit-deals-offer.com' is not in the merchant allowlist... the
transaction should be blocked."*

**Two small, honestly-reported quality wrinkles, neither serious enough
to fail the acceptance bar:**
- First run: one explanation's closing sentence said
  "warrants a HOLD_FOR_REVIEW" while the structured `recommendation`
  field said `BLOCK` — the narrative text and the structured output
  disagreed. Fixed by adding an explicit prompt instruction that the
  explanation's final sentence must match the `recommendation` field.
- Second run (post-fix): a different, rarer mismatch appeared instead —
  one session got `risk_level=LOW` paired with `recommendation=BLOCK`
  (internally odd, but erring toward caution, not under-caution; all
  underlying signals were genuinely low). This is a real small-local-model
  limitation, consistent with what Task 6/7 already found about
  `qwen2.5:3b-instruct` — not chased further with more prompt tuning
  since it didn't affect the acceptance-bar outcome (12/12 either way)
  and diminishing returns apply.

Added `tests/test_llm_verdict.py` (5 tests, pure-logic only — no live
LLM calls, matching this repo's existing convention: `consistency_check()`
tested against fixed signal combinations, plus a test that
`predict_all_merchants()` covers every real fraud-ring domain from
Task 8's network). Full suite: **46/46 passing**.

**Task 9 is complete.** Reproduce with `python3 -m defend.llm_verdict`.
Next per `plan.md`: Task 10, the Mutator — the closed-loop piece that
feeds Defend's false negatives back into Generate/Identify, the last
task in the build order.

## 2026-08-20 — Task 10 complete: Mutator closed loop (final task in plan.md's build order)

Reused every existing building block rather than reimplementing:
`cross_validated_predictions`/`build_feature_matrix` (Task 5),
`_mandate`/`_base_session` (Task 2's `narrative_generator.py`),
`_load_ieee_cis` (Task 4), `score_sessions`/`compute_constraint_drift`/
`compute_ingestion_source_trust_score` (Tasks 1/3). `mutator/mutate.py`'s
only new logic is the mutation prompt, the round loop, and honest
per-round reporting.

Checked what Task 5's model actually misses before designing anything:
reran `cross_validated_predictions` on the real 210-session dataset —
**6 hard cases** (false negatives + low-confidence catches, probability
< 0.65), all subtle hijacks, matching Task 5's already-reported subtle
recall (0.914 → ~6/70 missed).

**Three real bugs hit and fixed while building this — this task exercised
the pipeline in ways every prior task hadn't, and found real gaps:**

1. **Mutation prompt didn't tell the LLM the actual mandate cap.** First
   attempt asked for "a harder variant... reduce the amount deviation"
   without stating what the cap actually *was*. The LLM produced a
   variant with a **bigger** deviation (20x cap vs. the original's 4x) —
   the opposite of the goal. Fixed by passing the real
   `mandate.amount_cap`/`merchant_allowlist` into the prompt explicitly.

2. **Over-corrected the same prompt** on the second attempt — telling
   the LLM to use a "smaller multiple of the cap" produced sessions
   **under** the cap entirely (INR 1500 vs. a 3000 cap), which aren't
   violations at all anymore, just legitimate-looking sessions
   mislabeled as hijacks. Fixed with an explicit constraint: the signed
   amount must still exceed the cap (still a real violation), just by a
   smaller margin (e.g. 1.2x-1.8x instead of a large multiple) — subtler,
   not compliant.

3. **A real, general bug in Task 5's original `defend/lightgbm_baseline.py`**,
   exposed only once fresh IEEE-CIS rows were sampled: `_CATEGORICAL_COLUMNS`
   was a hardcoded list that happened to work on the original 210-row
   sample only because two IEEE-CIS identity columns (`id_23`, `id_27`)
   were all-NaN in that particular sample (IEEE-CIS identity data is
   very sparse — most transactions have none) — never showing up as a
   non-numeric dtype there. The Mutator's fresh sampling immediately hit
   real string values in those same columns, crashing LightGBM ("pandas
   dtypes must be int, float or bool"). Fixed generally, not by adding
   two more names to the list: `build_feature_matrix` now casts *any*
   remaining non-numeric column to `category`, not just ones on a fixed
   list. First attempt at this fix used `dtype == object`, which
   silently doesn't work in this pandas version (3.0.5 defaults string
   columns to its own `StringDtype`, not legacy `object`) — caught by
   writing a real regression test with a small fixture rather than
   trusting that the fix worked because a live run happened to pass;
   switched to `pandas.api.types.is_numeric_dtype` instead, which is
   robust to the dtype-inference distinction.

**Real per-round result** (`python3 -m mutator.mutate`, cumulative
dataset, 5-fold CV each round, same methodology as Task 5):

| Round | n | Precision | Recall | F1 | FP-rate (benign) |
|---|---|---|---|---|---|
| 1 (baseline) | 210 | 0.993 | 0.957 | 0.975 | 0.014 |
| 2 (+12 harder sessions) | 222 | 0.993 | 0.987 | 0.990 | 0.014 |
| 3 (+4 harder sessions) | 226 | 0.981 | 0.987 | 0.984 | 0.043 |

**Honest read, not smoothed into a curve that isn't there:** Round 1→2
is a real, meaningful improvement — recall rose from 0.957 to 0.987, and
4 of the original 6 hard cases stopped being hard at all after
retraining (only `agent-ee8c04d6`/`agent-e07cc46b` remained hard going
into Round 3). **Round 2→3 recall did not improve (flat at 0.987), and
the false-positive rate on benign traffic roughly tripled (0.014 →
0.043)** — both new rounds' own harder sessions were caught with 100%
recall once added to training data (the model *can* learn each specific
pattern immediately), but Round 3's additions came at a real precision
cost that Round 2's didn't. Reported plainly, per `plan.md`'s kill
criteria: this is not "round 2 doesn't improve, say so" territory
(round 2 clearly did improve) but it is exactly "don't claim an
improvement curve that isn't there" for round 3 specifically — the
diminishing-and-reversing pattern is the honest finding, not glossed
over as still-improving.

**Identify write-back**: 16 new `AttackVector` entries appended to
`identify/taxonomy.json` with `reviewed=False` (12 from round 2, 4 from
round 3) — taxonomy now has **77 entries total, 16 unreviewed**. Added
the `reviewed` field to `identify/schema.py`'s `AttackVector` as a
purely additive change (defaults `True` via `from_dict`, so the original
61 entries round-trip unchanged — verified with a regression test, not
assumed).

Added `tests/test_mutate.py` (3 tests: `find_hard_sessions` against a
fixture with a hand-crafted predictive feature so specific hard/easy
rows can be asserted on rather than fighting LightGBM randomness, plus
`reviewed`-field defaulting both ways) and
`tests/test_lightgbm_baseline.py` (2 tests, the dtype-safety-net
regression). Full suite: **51/51 passing.**

**Task 10 is complete — but this is NOT the last task in `plan.md`'s
build order.** Correcting an inaccurate claim made earlier in this log
(Tasks 8/9's entries called Task 10 "the last task"): `plan.md` §4 also
has **Task 11 (Web dashboard)** and **Task 12 (Integration contract +
latency measurement)** after it — neither built. Task 10 does close out
the three-pillar + closed-loop *system* plan.md's Part 1 describes, and
is the last of the ML/data/detection tasks, but the two remaining tasks
are real, undone work, not optional polish to wave past. Reproduce Task
10 with `python3 -m mutator.mutate` (resumable-safe: a mid-run failure
leaves `data/joined_sessions.csv` untouched since it's only written at
the very end, though `data/mutator_sessions.json` and
`identify/taxonomy.json` can accumulate partial state from a crashed
attempt — check and truncate those before rerunning, as happened twice
here while finding the bugs above).

**Project-wide status after all 10 tasks:** the confirmed primary Defend
signal is `constraint_drift`/`ingestion_source_trust_score` (not the
originally-planned `utterance_artifact_divergence`, which fails on
subtle attacks — see the Decision Log). All three required deliverable
pillars have real, runnable code: Identify (`identify/discover.py`,
LangGraph + Qdrant + networkx, 77 taxonomy entries), Generate
(`generate/`, 226 real LLM-generated + mutated sessions across two
attack types), Defend (5 layers: rules, LightGBM, content/semantic, GNN,
LLM verdict, all with real measured metrics). The closed loop is real,
not just a diagram: Task 10's Mutator genuinely feeds Defend's misses
back into both Generate (harder sessions) and Identify (new taxonomy
candidates), and the round-over-round table is measured, not assumed.
Remaining flagged-but-not-done item: `TEAM_BRIEF.md` §4.3's positioning
language still calls divergence "the core novelty claim," which the
Decision Log identifies as needing an update to match the confirmed
pivot — not yet done.

## Task 11 — Web dashboard (complete)

Built `dashboard/` (FastAPI + plain HTML/CSS/JS, no build step, per
`plan.md`/`TEAM_BRIEF.md` §5.3's "minimal frontend"/"ugly is fine"
framing): `dashboard/data_access.py` (all real-pipeline wiring, no new
logic), `dashboard/main.py` (routes + startup precompute), and
`dashboard/static/{index.html,app.js,style.css}` (Sessions view with
client-side filters, Session Detail view with a live "Get AI Verdict"
button, Metrics view with an inline SVG line chart — no charting
library).

Every number shown is real, computed at server startup or on demand,
never hardcoded for the UI:
- **`GET /api/sessions`** — all **226** sessions (210 from Task 2 +
  16 from Task 10's Mutator, deduped by `agent_id`), each with
  precomputed rules/constraint-drift/ingestion-trust/content/GNN
  signals. Verified live: `agent-3612bb96` (an obvious hijack) showed
  `rules=flagged, lightgbm=1.000, gnn=0.989` in the running dashboard,
  matching what Tasks 3/5/8 already measured for that class of session.
- **`POST /api/sessions/{agent_id}/verdict`** — calls
  `defend.llm_verdict.verdict_for_session()` live. Fixed a real bug this
  surfaced: that function only resolved sessions from
  `load_cached_dataset()` (the base 210), so any of Task 10's 16
  Mutator-hardened sessions would `KeyError` on this path — now merges
  in `mutator.mutate._load_mutator_cache()` too. Verified live in the
  browser: requesting a verdict for `agent-3612bb96` returned
  `HIGH / BLOCK` with an explanation correctly citing the amount/domain
  mismatch, the 1.000 classifier score, and the 0.989 GNN score.
- **`GET /api/metrics`** — Task 5's `cross_validated_predictions`
  re-run at startup (overall precision 0.981, recall 0.987, F1 0.984,
  AUC 0.992 on the live run — small day-to-day float noise vs. the
  numbers quoted earlier in this log, same methodology), Task 8's
  `train_and_evaluate()` re-run at startup (held-out precision 0.968,
  recall 1.000, F1 0.984, AUC 1.000), and Task 10's exact 3-round table
  read from `data/mutation_rounds.json` (added a `json.dump` at the end
  of `mutator/mutate.py`'s `run()` — it previously only printed this).

Second real bug found and fixed via the new test file: `compute_attack1_metrics`'s
`_slice()` helper called `.sum()` on a plain Python `list` mask for the
"overall" slice (only the `obvious`/`subtle` slices passed a numpy/pandas
boolean array) — `AttributeError` on the very first call. Fixed by
normalizing every mask through `pd.Series(...).values` before use.

Also fixed a mismatched label: `data_access.py` initially exposed each
session's mandate *category* (e.g. "travel") under a field named
`domain`, colliding with the actual URL-domain concept `extract_domain()`
uses for GNN lookups. Renamed to `category` throughout
(`data_access.py`, `app.js`) before this caused confusion in the UI.

`tests/test_dashboard_data_access.py` — 5 tests, deliberately exercising
the real pipeline (LightGBM CV + GNN training) once via a module-scoped
fixture rather than mocking, mirroring `tests/test_gnn.py`'s existing
style: dedup-by-agent_id, row-shape/signal-range checks over all 226
real sessions, attack-1 metrics shape, and the mutation-rounds file
(skips rather than fabricates if the file isn't present in a fresh
clone). Full suite: **56/56 passing.**

Verified end-to-end in the actual browser (not just curl): loaded the
Sessions view (226 rows, filters work), opened a hijacked session's
detail view (raw utterance / signed artifact / hidden injection payload
/ all instant signals rendered correctly), clicked "Get AI Verdict" and
got a real Ollama-generated HIGH/BLOCK verdict back, and loaded the
Metrics view (real per-attack tables + the 3-round SVG chart, screenshot
confirmed it actually draws three colored lines, not just text).

Run it: `python3 -m uvicorn dashboard.main:app --host 127.0.0.1 --port 8000`,
then open `http://127.0.0.1:8000`.

## Task 12 — Integration contract + latency measurement (complete)

Built `defend/integration_contract.py` + `INTEGRATION_CONTRACT.md`, per
`plan.md`'s exact file list. Defines the JSON payload the session-risk
score emits into a Decision Intelligence-style consumer
(`TEAM_BRIEF.md` §3.3/Part 9): runs at the agent-provider/PSP layer
(where the raw utterance lives, which the Mastercard network never
sees), emits a score that rides along with the signed Intent Artifact
(`intent_artifact_hash` in the payload) — DI consumes it, doesn't
compute it.

**Deliberate design decision, not in `plan.md`'s literal text but
following its own architecture directly**: the payload is built from
only the four fast, deterministic/ML Defend layers (rules, LightGBM,
content, GNN) — Task 9's LLM narrative verdict is excluded on purpose.
An LLM call doesn't belong in a per-transaction latency budget or in
front of a machine consumer that only needs a number; that layer stays
on-demand for a human analyst (the dashboard's "Get AI Verdict"
button). `session_risk_score = max(lightgbm_prob, gnn_prob)` — either
attack type should be able to raise risk — with `risk_level` thresholds
(0.5 HIGH, 0.15 MEDIUM) reused from `defend/llm_verdict.py`'s existing
`consistency_check()` rather than inventing a second cutoff scheme.

**Real measured latency** (`measure_latency()`, one LightGBM model fit
once and held in memory — the realistic production shape, not refit per
request — 50 single-session payloads timed individually end-to-end,
first call excluded as warm-up):

| Metric | Value |
|---|---|
| mean | 9.08 ms |
| p50 | 8.88 ms |
| p95 | 10.59 ms |
| max | 10.74 ms |

**Honest caveat surfaced, not hidden**: a real clean example session
(`agent-cfb4637f`) came back `risk_level=MEDIUM` rather than `LOW`,
purely because of `gnn_prob=0.2057` — the GNN's raw probability for a
merchant that's neither a labeled fraud-ring member nor one of the 19
hand-labeled legitimate merchants doesn't reliably sit near zero. This
is a real limitation of Task 8's GNN as trained, documented in
`INTEGRATION_CONTRACT.md` next to the example rather than swapped out
for a cherry-picked one that hides it.

`tests/test_integration_contract.py` — 5 tests: pure-logic threshold
checks on the extracted `_classify_risk_level()` helper, JSON-Schema
conformance validated with `jsonschema.validate()` (not just asserted by
eye) against a real payload, a hijacked-session sanity check, and a
latency-shape check. Full suite: **61/61 passing.**

**This closes out every task in `plan.md`'s build order — Tasks 1
through 12, all with real measured results, none mocked or
placeholder.**

## Doc pass — `TEAM_BRIEF.md` signal-pivot language updated (2026-08-20)

The one remaining flagged-but-not-done item from every "Tasks 1-12
complete" note above is now done: `TEAM_BRIEF.md` no longer calls
`utterance_artifact_divergence` "the core novelty claim." Updated, all
in `TEAM_BRIEF.md`:
- §4.3: renamed the section to `constraint_drift` + `ingestion_source_trust_score`,
  rewrote it from a forward-looking contingency ("unproven until
  tested... if it doesn't work, pivot") into the confirmed outcome (real
  AUC numbers: divergence subtle=0.419, constraint_drift subtle=0.979,
  ingestion_source_trust_score subtle=0.905), and swapped the inline
  schema comments marking which fields are the headline vs. fallback
  signal.
- Part 8 kill-criteria table: annotated the Aug 18 row **TRIGGERED**
  rather than leaving it as an unresolved contingency.
- Part 9's real-world-feasibility demo beat and Part 10's taxonomy
  entry #47: both updated to lead with constraint drift, replacing the
  `X ms` latency placeholder with Task 12's real measured number (mean
  9.08 ms, p95 10.59 ms).
- Part 5.4 demo script beats 3 and 8: reworded so the scripted
  explanation of *why* a hijack gets flagged matches what the shipped
  system actually flags on (constraint drift + content layer), not the
  originally-planned divergence-led story.
- "HONEST CLOSING NOTE": rewritten from a pre-build risk warning into a
  retrospective — states plainly that the contingency it warned about
  did happen, and what was actually done about it.

No numeric threshold or scope decision changed — this was a
find-and-update pass on stale prospective language now that the
outcome is known, not a new decision. `plan.md`/`task.md` were already
correct on this; only `TEAM_BRIEF.md`'s positioning prose was behind.

## `requirements.txt` — pinned versions, real dependency conflict found and fixed (2026-08-20)

Asked to check `requirements.txt` for proper versioning — it had none:
every line was a bare package name. Two real problems, not just missing
pins:

1. **Five direct imports had no matching line at all**: `generate/llm_adapter.py`
   imports `dotenv`, `google.genai`, `groq`, and `requests`; `defend/gnn.py`
   imports `torch` directly (not just via `torch_geometric`). All five
   were installed and working locally but absent from the file — a fresh
   clone running `pip install -r requirements.txt` would have hit
   `ModuleNotFoundError` on the very first LLM call or GNN run.
2. **A real, fresh-install-breaking version conflict**: pinning to the
   exact versions installed here (`pip freeze`) put `langchain==0.2.5`
   and `langgraph==1.2.11` in the same file. `pip install --dry-run`
   caught what nothing else in this environment had: `langgraph`
   requires `langchain-core<2,>=1.4.7`, `langchain==0.2.5` requires
   `langchain-core<0.3.0` — mutually exclusive. This environment only
   "worked" because `langchain` 0.2.5 and `langchain-core` 1.6.0 were
   installed via separate, unresolved `pip install` calls at different
   points this session, not because the combination is actually valid.
   `identify/ingestion/chunker.py`'s only use of the full `langchain`
   package is `RecursiveCharacterTextSplitter` — re-exported from the
   much lighter `langchain-text-splitters` package. Swapped the import
   to `from langchain_text_splitters import RecursiveCharacterTextSplitter`,
   upgraded that package to `1.1.2` (the first version whose own
   `langchain-core` constraint, `>=1.2.31,<2.0.0`, is actually compatible
   with `langgraph`'s), and dropped the `langchain` meta-package
   entirely — confirmed nothing else in the repo imports it.

Verified `pip install --dry-run -r requirements.txt` resolves cleanly
(no conflicts) after the fix, and the full suite still passes:
**61/61.**
