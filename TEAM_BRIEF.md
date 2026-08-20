# Mastercard Innovation Challenge — Team Brief (Final, Locked)
### Self-contained. Read fully before writing code.

> 🚨 **REGISTRATION CLOSES AUG 20.** Confirm every teammate is registered on the team before Thursday. Nothing in this document matters if the team isn't registered.

---

# PART 1 — POSITIONING (this is the whole submission)

## 1.1 What NOT to say

**Do NOT say:** "agentic commerce is an unsolved gap." Mastercard shipped Agent Pay in 2025; it's broadly available through certified processors in 2026, runs OpenAI Instant Checkout, and Verifiable Intent pilots began Feb 2026. A luxury watch showing up in a camping-supplies cart already gets blocked by their own system.

> ⚠️ **Never claim we detect "cart doesn't match intent."** That IS Verifiable Intent. Claiming it as novel gets us stopped in minute one — and "camping supplies" is Mastercard's own documentation example. Use a different domain in the demo.

**Everyone reads mastercard.com's Agent Pay page tonight. 15 minutes.** You cannot pitch against a product you haven't seen.

## 1.2 THE GAP — our entire submission

> ### Verifiable Intent proves the cart matches the mandate. Nothing proves the mandate matches the human.

Verifiable Intent operates **downstream** of the Intent Artifact. If the agent is hijacked by prompt injection **before** the artifact is signed, the artifact itself encodes the attacker's goal. Then:

- Cart matches intent ✅
- Signature chain valid ✅
- Agentic Token in scope ✅
- Verifiable Intent semantic check passes ✅
- **Money is gone.**

The security literature names this precisely: **execution integrity is preserved, but decision integrity is compromised.** Injection operates pre-signature, shaping how the decision gets made without ever violating a cryptographic guarantee. Put simply: you don't tamper with the mandate after signing — you tamper with what the agent believes the intent was before it picks up the pen.

**This is a seam Verifiable Intent structurally cannot reach**, because VI validates artifact→cart, and the corruption happens upstream, at human→artifact.

## 1.3 Prior work (cite it — don't get caught not knowing it)

- **Greshake et al. (arXiv 2302.12173, 2023)**, "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection" — the foundational paper naming indirect prompt injection as an attack class (adversary plants instructions in content the LLM retrieves, not text the user typed; demonstrated against Bing Chat and GPT-4 across data theft, worming, availability disruption, and ecosystem contamination). Our pre-signature corruption is this exact mechanism, applied specifically to the moment before a payment mandate is signed — cite this as the general precedent, 2601.22569 below as the AP2-specific one.
- **arXiv 2601.22569** (May 2026), "Whispers of Wealth: A Systematic Red-Teaming Study of AP2" — establishes the pre-signature injection threat model. **They red-team; they don't build detection.** We build the detection + closed loop on top. Cite as validation and extension.
- **Cloud Security Alliance** — AP2 threat framework (service-mesh poisoning, HITL fatigue, anomaly flooding)
- **Halborn** — poisoned tool outputs (spoofed price oracles/address books), context-window manipulation burying mandate constraints in long sessions
- **AP2 docs** — mandates bind to the *user's* signing key, not the agent's; a compromised agent can still yield a validly-signed malicious mandate; no in-protocol revocation of an Intent Mandate before TTL expiry.

## 1.4 Evidence (industry stats to cite)

- Ravelin's survey of 1,504 fraud professionals: 44% of enterprise merchants are already integrating agentic protocols, another 32% within six months — adoption outpacing safe management.
- Darwinium's survey of 500 fraud/risk leaders: 75% estimate more than a quarter of current fraud attempts are AI-assisted, yet 64% can only stop fraud at one or a few checkpoints.
- 50% of merchants rely on five to six separate fraud vendors — every handoff between tools is a gap AI-powered attacks will exploit.

That last stat is your "why one closed-loop system, not a stack of point tools" argument, handed to you by the industry itself.

---

# PART 2 — WHAT WE'RE ACTUALLY BUILDING

## 2.1 Two attacks, not three, not forty-six

| # | Attack | Role |
|---|---|---|
| 1 | **Pre-signature Intent Artifact corruption** — indirect prompt injection poisons an agent's context *before* it signs the Intent Artifact | **PRIMARY** — headline demo |
| 2 | **Transaction laundering / fraudulent merchant network** — the shell-merchant structure the hijacked purchase pays into: shared acquirers, shared beneficial ownership, rapid-registration domains, funnel storefronts | **SECONDARY** — graph layer |

> ⚠️ **Card rail, not account rail.** A hijacked agent *buys* something — the money goes to a **merchant** via an **acquirer**, not through a 4-6 hop account-to-account chain (that's a UPI/account-rail pattern and doesn't belong here). The GNN models merchant→acquirer→beneficiary, not account→account. The typosquat storefront the hijacked agent buys from *is a node in this network* — that's what connects attack #1 to attack #2 on the same rail.

## 2.2 We DOCUMENT 48 attacks, we BUILD 2

Taxonomy across 11 categories, 5 regions — including the two new precisely-worded entries added specifically to match what's demoed (see Part 10). Say the "48 documented, 2 built end-to-end" line out loud in the deck. Judges trust teams that draw an honest line between research breadth and build depth.

## 2.3 What we are explicitly NOT building

Say this out loud so nobody relitigates it on day 8:
- Not legitimate-session / digital-arrest social-engineering fraud (taxonomy entry only)
- Not RL-based mutation (confidence-guided LLM mutation instead — lower non-convergence risk)
- Not federated learning, not LLM-GNN token fusion, not foundation-model pretraining (future-work slide only — never claim as a built differentiator)
- Not three attack types. **Two.**

## 2.4 The three required deliverables

The challenge requires **all three**. Missing one invalidates the submission.

1. **Code repository** — runnable, documented, covers Identify + Generate + Defend
2. **Solution walkthrough** — deck (.pptx or .pdf)
3. **Working web prototype** — presentable, not beautiful

---

# PART 3 — TEAM & PREREQUISITES

## 3.1 Skillset fit
RAG systems, agentic AI, distributed job orchestration, FastAPI/Postgres full-stack, prior work on prompt-injection defenses and confidence calibration all map directly onto this build — this isn't a new stack, it's a scoped-down version of things already built before.

## 3.2 Knowledge prerequisites (skim before Day 1)
- **Everyone**: Mastercard's Agent Pay page, 15 minutes.
- **Person A**: CTGAN docs, OWASP LLM Top 10 (LLM01 — prompt injection).
- **Person B**: LightGBM tutorial if unfamiliar; PyTorch Geometric intro for the GNN.

## 3.3 Deployment layer — where does this actually run?

`utterance_artifact_divergence` (below) needs the **raw human utterance**. Mastercard's network never sees that — the **agent provider** does. So if a judge asks "how would this deploy on our network?", the honest answer is: **it can't run at the network layer.**

That's fine, stated first: this runs at the **agent-provider or PSP layer**, where the utterance lives, and emits a **session-risk score that rides along with the Intent Artifact** into the network. Mastercard's Decision Intelligence consumes the signal; it doesn't compute it. No new network infrastructure required — this is a stronger feasibility story than it first sounds, because it slots into existing Intent Artifact plumbing.

---

# PART 4 — ARCHITECTURE

## 4.1 System diagram

```
┌──────────────┐   attack schema    ┌───────────────┐
│   IDENTIFY   │ ─────────────────► │   GENERATE    │
│ (RAG agent + │                    │  (red team)   │
│  taxonomy)   │                    └───────┬───────┘
└──────┬───────┘                            │ agent sessions + transactions
       ▲                                    ▼
       │                            ┌───────────────┐
       │                            │    DEFEND     │
       │                            │  (blue team)  │
       │                            │ rules→LGBM→   │
       │                            │ content→GNN→  │
       │                            │ verdict       │
       │                            └───────┬───────┘
       │                                    │ misses + confidence scores
       │                            ┌───────▼───────┐
       └────────────────────────────│   MUTATOR     │──► harder v2 attacks
              new-attack candidates │ (confidence-  │    back to GENERATE
                                    │  guided LLM)  │
                                    └───────────────┘
```

**The feedback arrow is the differentiator.** Most teams build three disconnected components. This one closes the loop three ways — Defend's misses feed both a harder Generate round *and* the Identify agent as candidate new attack classes — and gets harder to fool each round, with the curve to prove it.

## 4.2 Identify pillar — mandatory, must be runnable code (not just a document)

The brief requires all three pillars as running code — a static taxonomy document alone fails this. Build a small RAG-based attack-discovery agent:

```
identify_agent/
  corpus/              # ~30-40 curated threat reports (Mastercard, Visa, Darwinium,
                        #   Ravelin, Zscaler, arXiv 2601.22569, CSA, Halborn)
  discover.py          # RAG retrieval → LLM extraction → structured attack entry
  schema.py            # AttackVector dataclass
  taxonomy.json        # 48 entries, machine-generated + human-curated
  novelty_score.py     # dedupe/rank new candidates vs. existing taxonomy
```

This reuses your existing `production_rag` pipeline, scoped down. Two payoffs beyond compliance: it closes the loop three ways (see 4.1), and it's a live demo beat — "here's the agent ingesting a threat report published last week and proposing a new attack vector our taxonomy didn't have."

## 4.3 Agent-session schema (LOCK DAY 2 — no changes after)

IEEE-CIS has no concept of an agent session. We construct one and join it:

```python
agent_session = {
    "agent_id": str,
    "agent_registry_status": str,        # mirrors Agentic Token validity
    "mandate_scope": {
        "categories": list,
        "amount_cap": float,
        "merchant_allowlist": list,
        "expiry": datetime,
    },
    "intent_artifact_hash": str,          # mirrors Verifiable Intent
    "raw_utterance": str,                 # ephemeral only — see 4.5 privacy note
    "task_origin_url": str,
    "content_sources_ingested": list,
    "utterance_artifact_divergence": float,  # secondary signal — catches obvious hijacks only, see 4.3
    "constraint_drift": float,            # ← PRIMARY SIGNAL (confirmed, see 4.3)
    "ingestion_source_trust_score": float,  # ← PRIMARY SIGNAL (confirmed, see 4.3)
    "hops_since_intent": int,
    "tool_calls_made": int,
    "injection_present": bool,            # label
}
# JOIN → IEEE-CIS transaction row (amount, merchant, device, timing, ...)
```

### `constraint_drift` + `ingestion_source_trust_score` — the confirmed primary signal (pivoted from `utterance_artifact_divergence`)

**Update, post-testing (see `task.md`'s Decision Log for the full record): the pivot described below as a contingency actually happened, and it's not a fallback anymore — it's the primary signal.**

`utterance_artifact_divergence` (semantic distance between the **raw user utterance** and the **signed Intent Artifact text** — the feature Agent Pay's token layer structurally cannot compute, since tokens verify scope at authorization time, not divergence between what the human said and what got signed) was prototyped and tested on the real 210-session generated set: `OBVIOUS AUC=1.000`, but `SUBTLE AUC=0.419` — barely above chance. Keeping the attacker's wording close to the original request (the realistic, subtle attack) makes the hijacked artifact read as semantically similar to the legitimate one, which is exactly the blind spot a pure semantic-distance signal has.

`constraint_drift` + `ingestion_source_trust_score` — checking the mandate cap and merchant allowlist directly, and the provenance of ingested content, rather than semantic similarity — don't share that blind spot: `OBVIOUS AUC=1.000`, `SUBTLE AUC=0.979` and `0.905` respectively. These are now the **primary Defend signal** and the headline claim for this submission. `utterance_artifact_divergence` isn't abandoned — it still separates obvious topic-swap hijacks perfectly and is kept as a secondary signal — but it is no longer the feature this submission leads with.

## 4.4 Privacy note (prepare this answer — payments-adjacent submissions get asked)

`utterance_artifact_divergence` requires processing the raw utterance at the agent-provider layer. Prepared line: **ephemeral processing only, no persistent storage of raw request text** — the divergence score is computed and retained; the utterance itself is not.

## 4.5 Sandbox note (prepare this answer)

If asked "did you test against a real Agent Pay/AP2 sandbox?": **no — simulated against the published AP2 mandate schema**, not a live sandbox, in a 12-day window. Say this unprompted in the deck rather than waiting to be asked.

## 4.6 Defend stack (build in this order)

| Layer | What it catches | Risk |
|---|---|---|
| 1. **Rules** | Mandate-scope violation, amount cap breach, merchant-allowlist miss, agent-registry status | Low |
| 2. **LightGBM** | Transaction-level anomaly on joined features | Low — **THIS IS THE FLOOR** |
| 3. **Content/semantic** | Injection payloads in ingested page content (reuse notification-router Jaccard + confidence-calibration pattern) | Medium |
| 4. **GNN** | Merchant→acquirer→beneficiary laundering topology (attack #2) | High — fallback: graph features (degree, hop-depth, community) fed into LightGBM |
| 5. **LLM verdict** | Fuses 1–4 into a plain-English explanation for a risk analyst | High — fallback: weighted ensemble + SHAP |

Layers 4 and 5 are optional. Layers 1–2 are not.

## 4.7 Fidelity validation

- Transaction features: **KS-test / Wasserstein distance** vs. real IEEE-CIS distribution, reported per feature, per attack type.
- Agent-session fields: validated for **behavioral plausibility** only (no public agent-session dataset exists to match against) — state this limitation honestly rather than implying a realism you don't have. Strengthen where possible by deriving mandate-scope/amount distributions from published agentic-commerce basket-size data or Agent Pay's own documented cap structures, rather than hand-picked values.

## 4.8 Metrics (every round)

- Precision, Recall, F1, AUC — **per attack type**, never blended.
- **False-positive rate on legitimate traffic** — every mutation round, non-negotiable. Recall gains that wreck precision are worse than useless in live payments.
- Per-round improvement curve across mutation rounds 1→2 (3 if time allows).

## 4.9 Confidence-guided mutation loop (the closed-loop novelty mechanic)

After round 1: take each false-negative session, prompt the narrative/CTGAN generator with "here's a pattern that evaded detection, generate variants that preserve the evasion property." Run 2 rounds minimum. Log precision/recall/F1 per round, per attack type — this curve is the demo centerpiece, the one thing a static submission literally cannot show.

---

# PART 5 — DELIVERABLES IN DETAIL

## 5.1 Code repository
- Runnable, documented, covers Identify + Generate + Defend.
- README with setup instructions, verified on a **clean clone** (Aug 29).
- Requirements pinned. Seed data or generation script included.

## 5.2 Deck (.pptx/.pdf) — slide order

1. The gap (§1.2 positioning paragraph)
2. Evidence (§1.4 stats)
3. Prior work (§1.3 — cite arXiv 2601.22569, CSA, Halborn)
4. Identify — RAG agent + 48-attack taxonomy + honest "2 built" line
5. Generate — simulation approach + fidelity metrics
6. Defend — five layers + per-attack metrics
7. Closed loop — three-way, mutation rounds, improvement curve
8. `constraint_drift` + `ingestion_source_trust_score` (confirmed primary signal, §4.3) — feature importance, why Verifiable Intent structurally can't compute either of these
9. Real-world feasibility — integration contract (JSON payload showing what the session-risk score emits into a Decision Intelligence-style consumer) + latency number (**measured: mean 9.08 ms, p95 10.59 ms, single session, all four Defend layers** — see `INTEGRATION_CONTRACT.md`)
10. Scoped out — what was deliberately not built, and why
11. Future work

## 5.3 Web prototype (live Aug 25)
- Agent session feed (clean + hijacked)
- Risk verdict panel (flag + reason + contributing signals)
- Metrics panel (per-attack, mutation curve)

## 5.4 Demo video (3 min, scripted, rehearsed)

**Do not use "camping supplies" — that's Mastercard's own example.** Use a different domain (e.g. travel booking).

1. **Legitimate** — "book me a flight to Bangalore under ₹8,000." Agent forms intent, signs artifact, books. Token valid, VI passes. ALLOW. *(Baseline — what Mastercard already handles correctly.)*
2. **Injected** — same request, but the agent reads a poisoned travel-aggregator page *before* forming intent. Injection reframes the task. Agent signs an artifact for "premium electronics, ₹4,00,000" — **and the cart matches that artifact perfectly.** Green checkmarks across the board: token valid ✅ signature valid ✅ cart-intent consistent ✅. *This is the moment the room understands the gap.*
3. **We flag it** — constraint drift catches the amount-cap and merchant-allowlist breach, content layer catches the injection phrasing: *"Signed intent diverges from user utterance: domain shift travel→electronics, 50× amount, artifact formed after ingesting 3 untrusted sources."* (Divergence spikes here too — this is an obvious hijack — but constraint drift is what still catches the subtle version of this attack, which divergence alone misses.)
4. **Follow the money** — GNN lights up the merchant/acquirer network the typosquat storefront sits in.
5. **The loop** — "Round 1 caught 78%. Fed the misses back. Round 2 is harder. Here's the curve."

**If only three beats work, ship 1–2–3.** That trio tells the whole story on its own.

---

# PART 6 — ASSIGNMENTS

**Person A — Red team (Identify + Generate)**
Agent-session schema · `utterance_artifact_divergence` prototype (Day 2, top priority in the whole project) · Identify RAG agent (reuses `production_rag`) · narrative generator (injection payloads: hidden CSS, JSON-LD poisoning, typosquat storefronts, poisoned tool output) · CTGAN transaction generator · fidelity validation · mutation loop

**Person B — Blue team (Defend)**
Rules layer · LightGBM ← **owns the Aug 23 gate** · content/semantic layer · GNN · LLM verdict (or SHAP fallback) · evaluation harness · all metrics · integration contract + latency measurement

**Person C** (or split A/B if 2-person team — both stop coding Aug 28)
Web dashboard from Aug 25 · deck · demo video · README + clean-clone reproducibility test

---

# PART 7 — SCHEDULE

Today is Mon Aug 17. Weekday evenings ~3–4 hrs. Weekends Aug 22–23 and Aug 29–30 full days.

| Date | Day | Work |
|---|---|---|
| **Aug 17** | Mon | Repo + README. Download IEEE-CIS, run EDA. **Write the gap statement (§1.2) into deck slide 1.** Everyone reads the Agent Pay page. |
| **Aug 18** | Tue | Lock agent-session schema. **Prototype `utterance_artifact_divergence` — does it separate? Flag tonight if not.** |
| **Aug 19** | Wed | Narrative generator v1 (~200 sessions, half injected). Start Identify corpus assembly. |
| **Aug 20** | Thu | Rules layer. Identify RAG pipeline (reuse `production_rag`). **Registration deadline — confirm today.** |
| **Aug 21** | Fri | Join agent-session rows to IEEE-CIS. `taxonomy.json` emitted by the Identify agent, not hand-written. |
| **Aug 22** | Sat (full) | CTGAN conditioned on attack label + fidelity checks (KS/Wasserstein). |
| **Aug 23** | Sun (full) | **HARD GATE: LightGBM with real precision/recall/F1/AUC. If not — stop all new features, ship baseline + Identify agent + simple UI. Still a valid, complete submission.** |
| **Aug 24** | Mon | Content/semantic layer. Integration contract JSON + latency measurement. |
| **Aug 25** | Tue | **Dashboard goes live.** Ugly is fine — wired to real model output tonight, gets 5 days of iteration. |
| **Aug 26** | Wed | GNN (merchant/acquirer/beneficiary graph). **If unstable tonight → graph features into LightGBM. Move on.** |
| **Aug 27** | Thu | Mutation rounds 1→2. Wire Defend's misses back into the Identify agent as new-attack candidates. Log FP rate every round. |
| **Aug 28** | Fri | LLM verdict agent, or SHAP fallback if not clean. **NO NEW CODE AFTER TONIGHT.** |
| **Aug 29** | Sat (full) | Polish. Deck finalized. Demo video recorded. Clean-clone reproducibility test. |
| **Aug 30** | Sun | **SUBMIT.** Do not wait for Aug 31. |
| **Aug 31** | Mon | Buffer only. |

---

# PART 8 — KILL CRITERIA (decided now, not at 2am)

| Date | If broken | Do this |
|---|---|---|
| Aug 18 | `utterance_artifact_divergence` doesn't separate clean from hijacked | Switch primary signal to `constraint_drift` + `ingestion_source_trust_score` (provenance-based) — **TRIGGERED: subtle-hijack AUC=0.419, switch made, see §4.3** |
| **Aug 23** | **No real LightGBM numbers** | **Stop all new features. Ship baseline + Identify agent + simple UI. This is still a complete, valid submission.** |
| Aug 26 | GNN unstable | Drop to graph features (degree, hop-depth, community) fed into LightGBM |
| Aug 27 | No measurable round-2 delta | Reframe headline claim: "round 1 demonstrated, closed loop designed" — never claim an improvement curve that doesn't exist |
| Aug 28 | LLM verdict agent messy | Weighted ensemble + SHAP feature importance — still explainable, never demo something half-working live |

---

# PART 9 — JUDGE Q&A (rehearse before Sep 8 GFF presentation)

| Question | Answer |
|---|---|
| **"Verifiable Intent already does this."** | VI validates artifact→cart. We validate human→artifact. If injection lands before signing, the artifact encodes attacker intent and the cart matches it perfectly — VI passes. Execution integrity holds; decision integrity is already gone. VI operates downstream of the corruption. |
| **"Doesn't Agent Pay solve agent fraud?"** | Agentic Tokens answer *is this agent registered and scoped*. VI answers *is the cart consistent with the recorded intent*. Neither answers *was the recorded intent what the human meant*. That's our layer, on top of theirs — complementary, not competing. |
| **"How is this different from Decision Intelligence?"** | DI scores transactions. We score agent sessions pre-signature — utterance divergence, ingestion provenance, constraint drift. Different feature space, designed to feed DI a session-risk signal. We emit a score that rides along with the Intent Artifact; you consume it, you don't compute it. No new network infrastructure required. |
| **"Did you test against a real Agent Pay/AP2 sandbox?"** | No — simulated against the published AP2 mandate schema in a 12-day window. Say this unprompted. |
| **"How do you handle the raw user utterance — privacy?"** | Ephemeral processing only. The divergence score is retained; the raw utterance text is not persisted. |
| **"Synthetic data — why believe the numbers?"** | Transaction features are fidelity-validated against real IEEE-CIS via KS/Wasserstein, per feature, per attack type. Agent-session fields are stated honestly as behaviorally plausible, not distributionally validated — no public agent-session dataset exists to check against. |

---

# PART 10 — TAXONOMY FIX (two entries added to match what's actually built)

The original 46-item taxonomy didn't contain entries matching either built attack precisely — a judge cross-checking the demo against the Identify document would find no numbered match. Two entries were added (taxonomy is now 48 items, 11 categories):

| # | Attack | Mechanism | Grounding | Defend signal |
|---|---|---|---|---|
| **47** | **Pre-signature Intent Artifact corruption** | Indirect prompt injection poisons an AI payment agent's context *before* it signs an Intent Artifact (AP2/Verifiable Intent). The resulting mandate is cryptographically valid and internally self-consistent — cart matches artifact, signature checks out — but the artifact encodes the attacker's goal, not the human's. Distinct from the taxonomy's pre-existing generic item #18 (post-authorization payment injection): this specifically targets the reasoning step *before* any cryptographic guarantee exists. | Greshake et al. (arXiv 2302.12173, 2023) establishes indirect prompt injection as a general attack class; arXiv 2601.22569 formally applies it to AP2 as the decision-integrity vs. execution-integrity gap; corroborated by Halborn and CSA's AP2 threat framework. | Constraint drift (mandate cap/allowlist checks) and ingestion-source trust score across the session — confirmed primary, see §4.3; semantic divergence between raw utterance and signed artifact text kept as a secondary signal (catches obvious hijacks, not subtle ones) |
| **48** | **Transaction laundering via fraudulent merchant network (card rail)** | Funds from a hijacked agent purchase flow into a shell-merchant network — sibling storefronts sharing an acquirer or beneficial owner, registered within days of each other, no independent trading history. Distinct from the taxonomy's pre-existing UPI/account-hop entries (#33–35), which are a different rail entirely. | Same graph-clustering approach top IEEE-CIS Kaggle solutions used (card+address+email-domain overlap) to detect merchant fraud rings in the original 2019 competition; broader academic precedent in GNN-based fraud-ring detection over transaction networks (e.g. arXiv 2307.05633, "Transaction Fraud Detection via an Adaptive Graph Neural Network"). Note: this grounding is thinner than #47's — no paper addresses this exact card-rail merchant-network setup directly, unlike #47's AP2-specific coverage. | GNN over merchant/acquirer/beneficiary graph — shared acquirer ID, domain-registration recency, card/address/email overlap clustering |

---

# HONEST CLOSING NOTE (updated post-build — see `task.md` for the full log)

`utterance_artifact_divergence` was written into this plan as the load-bearing wall, flagged explicitly as unproven pending the Day 2 isolation test. The test ran on the real 210-session generated set and the contingency this section warned about actually happened: `SUBTLE AUC=0.419` (barely above chance) — hijacked sessions that keep the wording close to the original request don't separate cleanly on semantic distance alone. Per this section's own instruction, the pivot was made rather than forced: `constraint_drift` + `ingestion_source_trust_score` are the confirmed primary Defend signal (§4.3), `utterance_artifact_divergence` is kept as a secondary signal for the obvious-hijack case it still handles perfectly. All 12 tasks in `plan.md`'s build order are complete on that basis, with real measured results throughout, not the divergence-led story this document originally assumed.

The schedule assumed evenings plus two full weekends with a functioning 2–3 person team. That held.

Before the team fully commits, have someone read Mastercard's own Verifiable Intent developer documentation directly, not secondhand — the positioning has been revised twice already after checking secondary sources first.
