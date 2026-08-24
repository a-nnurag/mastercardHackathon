I read the foundation context first, then inspected the repository inside the ZIP, including the source tree, current tests, duplicated package, generation/defense pipeline, dashboard, mutation path, and the existing contracts.

The important conclusion is: **the foundation commit should be a structural refactor, not a new-feature commit.** The current code already has most of the ingredients, but they are coupled in a way that makes Attack 3–6 difficult to add safely.

I would give the coding agent the following implementation plan.

---

# Foundation Commit — File-by-File Implementation Plan

## 0. Foundation target

The repository should end this commit with this logical flow:

```text
Raw Agent Session
       │
       ▼
Signal Detectors
 ├── constraint drift
 ├── source trust
 ├── content
 ├── divergence
 ├── LightGBM
 └── GNN
       │
       ▼
SignalSet / Evidence
       │
       ▼
Risk Engine
       │
       ├── hard security rules
       ├── risk fusion
       ├── missing-signal handling
       └── final decision
       │
       ▼
DecisionContract
       │
       ├── ALLOW
       ├── HOLD
       └── BLOCK
       │
       ▼
LLM Explanation
       │
       ▼
Audit / API / Dashboard
```

The critical invariant is:

> **The LLM can explain a decision, but it can never create, modify, lower, or override the decision.**

And:

> **Training happens offline. Runtime code only loads trained models and performs inference.**

Do **not** add Attack 3–6.

---

# 1. `generate/session_schema.py`

### Current problem

`AgentSession` currently mixes three different concepts:

```text
raw facts
+
derived detector outputs
+
ground-truth/evaluation state
```

For example:

```python
utterance_artifact_divergence
constraint_drift
ingestion_source_trust_score
```

are derived signals but live directly inside the raw session object.

This is the first thing to fix.

### Change

Keep `AgentSession` as the **input/raw session contract**.

It should contain things such as:

* `agent_id`
* `agent_registry_status`
* `mandate_scope`
* `intent_artifact_hash`
* `raw_utterance`
* `signed_artifact_text`
* `task_origin_url`
* `content_sources_ingested`
* `hops_since_intent`
* `tool_calls_made`
* `injection_payload_text`
* `injection_present` only as evaluation ground truth

Remove detector-produced values from the raw object:

```python
utterance_artifact_divergence
constraint_drift
ingestion_source_trust_score
```

### Add

Create a new immutable-ish signal container:

```python
@dataclass
class SignalResult:
    name: str
    value: float | bool | str | None
    available: bool
    evidence: list[str]
    metadata: dict
```

And:

```python
@dataclass
class SignalSet:
    signals: dict[str, SignalResult]
```

This becomes the canonical output of detection.

### Important

Do not rename existing raw fields unnecessarily. Existing generated datasets and tests depend heavily on them.

Also fix the existing schema mismatch discovered during inspection:

`generate/narrative_generator.py` passes:

```python
injection_payload_text=...
```

into `AgentSession`, but the current dataclass does not define it.

The foundation refactor must resolve this properly.

---

# 2. Create `defend/contracts.py`

This becomes the **central security contract**.

Do not scatter decision dictionaries across modules.

Create:

```python
@dataclass
class DetectionEvidence:
    signal_name: str
    value: ...
    confidence: float | None
    evidence: list[str]
```

```python
@dataclass
class RiskDecision:
    risk_score: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["ALLOW", "HOLD", "BLOCK"]
    signals: dict[str, ...]
    evidence: list[...]
    attack_family: str | None
```

The serialized contract should correspond to the foundation context's target:

```json
{
  "risk_score": 0.0,
  "risk_level": "LOW",
  "decision": "ALLOW",
  "signals": {},
  "evidence": [],
  "attack_family": "prompt_injection"
}
```

### Rules

* score always `[0, 1]`
* risk level always derived from the risk engine
* decision always derived from the risk engine
* attack family is metadata/classification, not an arbitrary LLM field
* evidence is append-only from detectors
* no detector can directly authorize a transaction

This file becomes the contract future AttackSurface implementations will consume.

---

# 3. Create `defend/signals/`

Create:

```text
defend/signals/
    __init__.py
    base.py
```

`base.py` defines the common detector abstraction.

For example:

```python
class SignalDetector(ABC):
    name: str
    attack_families: tuple[str, ...]

    @abstractmethod
    def detect(self, session: AgentSession) -> SignalResult:
        ...
```

The important abstraction is **signal detector**, not model.

That means a future Attack 3 can contribute signals without changing the risk engine.

---

# 4. `defend/constraint_drift.py`

### Current role

This is already one of the strongest signals according to the project's actual experiments.

Keep the underlying logic.

### Refactor

Convert the module from:

```python
compute_constraint_drift(session) -> float
```

being called ad hoc everywhere, to a detector implementation:

```python
ConstraintDriftDetector
```

It should still expose the existing pure function for backwards compatibility if useful:

```python
compute_constraint_drift(...)
```

but internally the runtime should use the detector interface.

### Also expose evidence

Instead of only:

```text
0.5
```

produce evidence such as:

```text
amount_cap_breached
merchant_allowlist_violation
```

with actual values where available.

Do not invent evidence when a value cannot be extracted.

---

# 5. `defend/rules.py`

Keep the deterministic rules.

But change their role.

### Current problem

`apply_rules()` returns:

```python
(bool, list[str])
```

and downstream code independently decides what that means.

### Change

Create:

```python
RuleEngine
```

or:

```python
RuleDetector
```

which returns structured evidence.

For example:

```text
rules.mandate_amount_violation
rules.merchant_allowlist_violation
rules.agent_registry_invalid
```

### Critical invariant

Rules are allowed to create **hard security constraints**.

The risk engine may therefore receive:

```python
hard_block=True
```

but this should still be represented through the common decision contract.

Do not let individual rule functions directly return `"BLOCK"`.

---

# 6. `defend/content_layer.py`

Keep the current Jaccard-based detector.

The foundation context explicitly says:

> improve content detection behind a clean detector interface; do not over-engineer it yet.

So do **not** replace it with another LLM/model.

Wrap it as:

```python
ContentInjectionDetector
```

Output:

```python
SignalResult(
    name="content_injection",
    value=score,
    ...
)
```

Preserve:

```python
REFERENCE_INJECTION_PHRASES
score_injection_likelihood()
```

for current tests/backwards compatibility.

---

# 7. `defend/divergence.py`

Keep this detector.

Do not remove semantic or lexical divergence.

But change its architectural status:

```text
secondary signal
```

not:

```text
final decision
```

The repository's own evidence says:

```text
obvious attacks → strong
subtle same-domain attacks → weak
```

So the implementation should make this explicit in naming/documentation.

Create:

```python
DivergenceDetector
```

returning:

* combined divergence
* lexical divergence
* semantic divergence
* availability/model metadata

Do not allow divergence to determine the final decision by itself.

---

# 8. `defend/lightgbm_baseline.py`

This stays.

**Do not delete LightGBM.**

But its responsibility must become very explicit:

```text
offline:
    build features
    train model
    evaluate model
    persist model

online:
    load model
    predict
```

### Split current responsibilities

Current:

```python
train_baseline(...)
```

is okay as training logic.

Add a persistence layer:

```python
save_model(model, path)
load_model(path)
predict(model, feature_row)
```

Do not train during prediction.

### Important

The feature matrix builder remains reusable:

```python
build_feature_matrix()
```

but it should not be responsible for training.

---

# 9. Create `defend/training/`

Use a small training package rather than putting all offline logic into runtime modules.

```text
defend/training/
    __init__.py
    lightgbm.py
    gnn.py
    pipeline.py
```

### `defend/training/lightgbm.py`

Responsible for:

* loading training dataset
* feature construction
* training
* validation
* metrics
* model persistence

### `defend/training/gnn.py`

Responsible for:

* graph construction
* train/test split
* GNN training
* evaluation
* model persistence

### `defend/training/pipeline.py`

One offline command that trains everything:

```bash
python -m defend.training.pipeline
```

It should generate model artifacts into a predictable directory such as:

```text
artifacts/models/
```

Do not commit generated model binaries unless the repository already intentionally tracks them.

---

# 10. `defend/gnn.py`

This is the **most important technical fix**.

### Current problem

`predict_all_merchants()` currently does:

```text
build graph
create model
TRAIN MODEL
predict
```

That violates the foundation requirement.

### Refactor into three responsibilities

```python
build_graph_data(...)
```

```python
train_gnn(...)
```

```python
predict_merchants(model, graph)
```

Prediction must never contain:

```python
optimizer = ...
loss.backward()
optimizer.step()
```

### New lifecycle

Offline:

```text
build graph
→ train GNN
→ evaluate
→ save model
```

Online:

```text
load graph/model
→ construct current graph representation
→ inference only
```

### Also preserve

```python
train_and_evaluate()
```

temporarily as a compatibility wrapper around the new training API if existing tests use it.

But make it clearly an **offline evaluation function**, not something used by the dashboard.

---

# 11. Create `defend/model_registry.py`

Keep this deliberately small.

Responsibilities:

```text
model name
model version
artifact path
training metadata
load model
```

Something conceptually like:

```python
ModelRegistry.load("lightgbm")
ModelRegistry.load("merchant_gnn")
```

The registry should not train anything.

This gives the runtime one clean way to obtain trained models.

Do not turn this into a large plugin system yet.

---

# 12. Create `defend/risk_engine.py`

This becomes the heart of the foundation commit.

### Input

```python
session
SignalSet
```

plus configured thresholds/policies.

### Output

```python
RiskDecision
```

### Responsibilities

1. collect available signals
2. handle missing signals
3. apply hard security violations
4. combine model/detector signals
5. normalize to `[0,1]`
6. determine `risk_level`
7. determine `ALLOW/HOLD/BLOCK`
8. attach evidence
9. attach attack family

### Important

Do not hardcode:

```python
max(lightgbm_prob, gnn_prob)
```

which the current integration contract does.

Instead use a generic fusion policy:

```python
RiskFusionPolicy
```

It should accept signal names/weights/thresholds rather than knowing that:

```text
LightGBM = attack 1
GNN = attack 2
```

This is what makes future attack plugins possible.

---

# 13. `defend/integration_contract.py`

This file currently contains too much architecture.

Current implementation:

```text
rules
→ LightGBM
→ content
→ GNN
→ max()
→ risk level
```

Refactor it to:

```text
session
→ detectors
→ SignalSet
→ RiskEngine
→ DecisionContract
→ serialized payload
```

### Preserve the external payload concept

Keep:

```python
SESSION_RISK_PAYLOAD_SCHEMA
```

but update it around the canonical `RiskDecision`.

The payload should expose:

```text
risk_score
risk_level
decision
signals
evidence
attack_family
```

rather than having separate ad hoc semantics such as:

```python
session_risk_score = max(lightgbm_prob, gnn_prob)
```

### Latency

Keep `measure_latency()`.

But it must measure:

```text
signal inference + risk fusion
```

not model training.

This makes the latency number meaningful.

---

# 14. `defend/llm_verdict.py`

This needs a major responsibility change.

### Current problem

The LLM is asked to produce:

```text
risk_level
recommendation
explanation
```

That violates the foundation architecture because the LLM can theoretically disagree with the deterministic/ML engine.

### Change

The LLM receives an already-finalized:

```python
RiskDecision
```

and only produces:

```python
Explanation
```

For example:

```python
@dataclass
class RiskExplanation:
    summary: str
    evidence_summary: list[str]
```

The prompt must explicitly say:

```text
The security decision is already finalized.
Do not change the decision.
Do not introduce new evidence.
Do not downgrade or upgrade the risk.
Explain only the supplied decision and evidence.
```

### Remove

LLM-generated:

```text
risk_level
recommendation
```

as authoritative fields.

If they remain in the UI, they must come from `RiskDecision`, not from the LLM.

---

# 15. `generate/generated_sessions.py`

Mostly preserve.

Change only what is necessary to support the new raw/derived separation.

The generated cache should persist **raw session facts**, not calculated detector outputs.

This is important because derived signals must be reproducible.

If the detector implementation changes:

```text
same raw dataset
→ recompute signals
```

rather than having stale signals stored in the session.

---

# 16. `generate/session_schema.py` serialization

Update:

```python
to_row()
```

so it clearly represents raw/session fields versus evaluation metadata.

Do not put new final decision fields into the generated session.

The generated session should never contain:

```text
risk_score
risk_level
decision
```

because those belong to Defend.

---

# 17. `generate/join_ieee_cis.py`

Keep this as the traditional-fraud grounding layer.

The foundation context explicitly says IEEE-CIS must **not** be represented as an agentic-fraud dataset.

Update comments/docstrings to make this distinction clear:

```text
IEEE-CIS = transaction/behavioral baseline
Synthetic AgentSession = GenAI attack simulation
```

Do not change the dataset into something it isn't.

---

# 18. `generate/merchant_network.py`

Keep the existing network generation.

No new attack family.

Only make sure the graph construction is usable by both:

```text
offline GNN training
```

and:

```text
online GNN inference
```

without the inference path retraining.

---

# 19. `mutator/mutate.py`

Do not redesign the mutation system in this commit.

Only update it to consume the new contracts where required.

Its current responsibility remains:

```text
find missed attacks
→ generate harder variants
```

Do not turn mutation into the foundation risk engine.

The future closed-loop architecture should remain possible, but foundation should not expand it.

---

# 20. `identify/`

### `identify/schema.py`

Keep `AttackVector`.

No major rewrite.

But document that this is the **attack-taxonomy contract**, separate from runtime detection contracts.

### `identify/discover.py`

No architectural rewrite.

It should continue producing taxonomy candidates.

### `identify/knowledge_graph.py`

No major rewrite.

### `identify/novelty_score.py`

No major rewrite.

### `identify/graph/*`

No major rewrite.

### `identify/vectorstore/*`

No major rewrite.

The Identify pillar already has a reasonable separation and isn't the source of the foundation problem.

---

# 21. `dashboard/data_access.py`

This currently does too much work at startup.

Especially:

```python
predict_all_merchants()
```

which currently trains the GNN.

After the foundation refactor:

```text
load model
→ infer
```

only.

### Change

Dashboard should consume:

```python
RiskDecision
```

and:

```python
SignalSet
```

instead of independently recomputing pieces of the security pipeline.

The dashboard becomes a consumer of the core runtime.

It must not have its own risk-fusion logic.

---

# 22. `dashboard/main.py`

Keep API endpoints broadly compatible.

Change `/verdict`:

```text
session
→ runtime
→ final decision
→ LLM explanation
```

not:

```text
session
→ LLM decides verdict
```

The returned JSON should expose:

```text
decision
risk_level
risk_score
signals
evidence
explanation
```

where explanation is the only LLM-generated portion.

---

# 23. Duplicate `mastercard_fraud_defense/` package

The ZIP contains a second stale copy:

```text
mastercard_fraud_defense/
    identify/
    defend/
    generate/
```

while the actual working repository also has:

```text
identify/
defend/
generate/
```

This is dangerous.

### Foundation action

Determine which package is actually imported/executed.

From the inspected code, the active modules consistently use:

```python
from defend...
from generate...
from identify...
```

Therefore the root-level packages are the real implementation.

The nested:

```text
mastercard_fraud_defense/
```

appears to be a stale duplicate.

### Recommendation

Remove the duplicate **only after verifying with imports/tests that nothing depends on it**.

Do not maintain two copies of the architecture.

If the coding agent finds an intentional packaging reason for it, document that instead of deleting it.

---

# 24. `INTEGRATION_CONTRACT.md`

Update this document after the code is refactored.

It should describe the new canonical contract:

```text
Session
→ Signals
→ RiskDecision
→ Explanation
```

Document explicitly:

```text
LLM is not authoritative.
```

And:

```text
training is offline.
```

Do not add future attack-family specifications here yet.

---

# 25. `README.md`

Only update the architecture/setup sections required by the refactor.

The README should now explain:

```text
1. Generate/load sessions
2. Train models offline
3. Runtime loads trained models
4. Signal detectors execute
5. Risk engine makes decision
6. LLM explains
```

Also document the training command and runtime command.

Do not rewrite the product positioning.

---

# 26. `CLAUDE.md`

Add the foundation invariants so future coding agents cannot accidentally undo the architecture.

The most important rules:

```text
RULE 1:
Raw session facts must not contain derived detector outputs.

RULE 2:
Detectors emit SignalResult/SignalSet.

RULE 3:
Only RiskEngine produces RiskDecision.

RULE 4:
LLM never produces or modifies the security decision.

RULE 5:
No model training in runtime/inference code.

RULE 6:
LightGBM remains a behavioral/transaction baseline until
evaluation proves otherwise.

RULE 7:
GNN inference must never train.

RULE 8:
IEEE-CIS is traditional transaction-fraud grounding, not
an agentic-fraud dataset.

RULE 9:
Attack 1 and Attack 2 must remain functional.

RULE 10:
Do not add Attack 3–6 during foundation work.

RULE 11:
New attack surfaces must eventually plug into the common
signal/attack abstraction instead of modifying RiskEngine internals.
```

This is important because your next parallel-development phase will depend on this file.

---

# 27. Tests — modify existing tests

Do not simply add a few happy-path tests.

The foundation commit needs tests proving the **architectural invariants**.

### `tests/test_session_schema.py` — new

Test:

* raw session serialization
* derived signals are separate
* round-trip persistence
* `injection_payload_text` works correctly

---

### `tests/test_signal_contract.py` — new

Test:

```text
detector → SignalResult
```

and validate:

* name
* value
* availability
* evidence

---

### `tests/test_risk_engine.py` — new

This is the most important new test file.

Test:

1. no signals → deterministic safe fallback
2. one signal → valid risk decision
3. multiple signals → fused decision
4. missing GNN → still produces decision
5. hard rule violation → cannot become ALLOW
6. score always `[0,1]`
7. risk level is consistent with score
8. attack family is preserved
9. evidence survives fusion

---

### `tests/test_llm_verdict.py`

Replace the current assumption that the LLM determines:

```text
risk_level
recommendation
```

with tests proving:

```text
RiskDecision is passed into explanation
```

and that the LLM output cannot change it.

A mocked LLM should attempt to return:

```json
{
  "risk_level": "LOW",
  "recommendation": "ALLOW"
}
```

for a BLOCK decision.

The final system must still return:

```text
BLOCK
```

---

### `tests/test_gnn.py`

Add the key regression test:

> Calling prediction must not invoke optimizer/training.

The simplest architectural test is to ensure the prediction API accepts a trained model rather than constructing/training one internally.

Keep the existing graph-shape tests.

---

### `tests/test_lightgbm_baseline.py`

Add:

```text
train → save → load → prediction consistency
```

Do not test exact floating-point equality if the backend makes that fragile; use a reasonable tolerance.

---

### `tests/test_integration_contract.py`

Rewrite around:

```text
session
→ signal set
→ risk engine
→ payload
```

and preserve JSON-schema validation.

Also ensure:

```text
decision != LLM output
```

is demonstrably true.

---

# 28. Add `tests/test_training_inference_boundary.py`

This should explicitly enforce the foundation rule:

```text
offline training ≠ online inference
```

Test that:

* LightGBM inference doesn't train
* GNN inference doesn't train
* risk engine doesn't train
* dashboard startup doesn't train
* LLM explanation doesn't train

This is worth having as a dedicated regression test because this exact lifecycle bug already exists in `gnn.py`.

---

# 29. `requirements.txt`

Do not introduce a large new dependency stack.

Only add dependencies if the refactor genuinely requires them.

The foundation architecture can be implemented using the current stack:

```text
dataclasses
LightGBM
PyTorch/PyG
sklearn
jsonschema
existing dependencies
```

Avoid adding:

* another orchestration framework
* another vector DB
* another LLM
* another graph database
* microservices
* message queues

---

# 30. Files that should NOT be changed materially

The coding agent should leave these essentially alone unless imports break because of the refactor:

```text
identify/corpus/*
identify/taxonomy.json
identify/knowledge_graph.json

identify/ingestion/*
identify/graph/*
identify/vectorstore/*

generate/merchant_network.py
generate/narrative_generator.py
generate/llm_adapter.py

mutator/mutate.py
dashboard/static/*
```

The foundation commit is not an opportunity to "clean up everything."

---

# Final target structure

After the foundation commit, I would want the core tree to look approximately like:

```text
MasterCard_fraud_Detection/
│
├── identify/
│   └── ... existing discovery system
│
├── generate/
│   └── ... existing generation system
│
├── defend/
│   ├── contracts.py
│   ├── risk_engine.py
│   ├── model_registry.py
│   │
│   ├── signals/
│   │   ├── __init__.py
│   │   └── base.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── lightgbm.py
│   │   ├── gnn.py
│   │   └── pipeline.py
│   │
│   ├── rules.py
│   ├── constraint_drift.py
│   ├── divergence.py
│   ├── content_layer.py
│   ├── lightgbm_baseline.py
│   ├── gnn.py
│   ├── llm_verdict.py
│   ├── integration_contract.py
│   └── evaluation.py
│
├── dashboard/
│
├── tests/
│   ├── test_session_schema.py
│   ├── test_signal_contract.py
│   ├── test_risk_engine.py
│   ├── test_training_inference_boundary.py
│   └── ... existing tests
│
├── artifacts/
│   └── models/              # generated, not source
│
├── CLAUDE.md
├── INTEGRATION_CONTRACT.md
└── README.md
```

---

# Execution order for the coding agent

I would **not** let the coding agent implement these files in arbitrary order.

Use this sequence:

```text
Phase 1
contracts.py
session_schema.py
tests for contracts/schema

        ↓

Phase 2
signals/base.py
refactor constraint_drift
refactor rules
refactor content_layer
refactor divergence

        ↓

Phase 3
LightGBM training/inference separation
GNN training/inference separation
model_registry.py

        ↓

Phase 4
risk_engine.py
risk-engine tests

        ↓

Phase 5
integration_contract.py
LLM explanation boundary

        ↓

Phase 6
dashboard integration

        ↓

Phase 7
training/inference boundary tests
full regression suite

        ↓

Phase 8
README / CLAUDE / integration contract
remove or quarantine duplicate package
```

---

## Foundation commit acceptance criteria

I would give the coding agent these **hard gates**:

### Architecture

* [ ] Raw `AgentSession` contains no detector-generated signal values.
* [ ] All detectors can emit the common `SignalResult`.
* [ ] `RiskEngine` is the only component producing `RiskDecision`.
* [ ] Risk fusion is generic and does not hardcode `LightGBM=max(GNN)` semantics.
* [ ] Missing detectors/signals are supported.
* [ ] Attack family is metadata, not hardcoded into the core engine.

### Model lifecycle

* [ ] LightGBM training is offline.
* [ ] GNN training is offline.
* [ ] GNN prediction never calls optimizer/training.
* [ ] Runtime loads trained artifacts.
* [ ] Dashboard startup performs inference only.

### Security

* [ ] Hard mandate violations cannot result in `ALLOW`.
* [ ] LLM cannot alter `risk_score`.
* [ ] LLM cannot alter `risk_level`.
* [ ] LLM cannot alter `decision`.
* [ ] LLM cannot introduce authoritative evidence.

### Existing system

* [ ] Attack 1 still works.
* [ ] Attack 2 still works.
* [ ] `constraint_drift` remains primary.
* [ ] `ingestion_source_trust_score` remains available.
* [ ] divergence remains a secondary signal.
* [ ] LightGBM remains in the system.
* [ ] IEEE-CIS remains a behavioral/transaction baseline.
* [ ] Identify still runs.
* [ ] Existing mutation path still works.

### Scope

* [ ] No Attack 3.
* [ ] No Attack 4.
* [ ] No Attack 5.
* [ ] No Attack 6.
* [ ] No new major ML model.
* [ ] No new infrastructure.
* [ ] No unnecessary rewrite of Identify/Generate.

---

### One important repository-specific issue

There is already a concrete bug that the foundation work should fix rather than blindly preserving: `generate/narrative_generator.py` constructs `AgentSession(..., injection_payload_text=...)`, while the current `AgentSession` definition does not declare that field. So the schema separation isn't merely theoretical—the current raw-session contract is already inconsistent with the generator.

The other major concrete defect is `defend/gnn.py`: `predict_all_merchants()` **trains a new GNN every time prediction is requested**, and both the dashboard and integration path call it. That is exactly the lifecycle violation identified by the foundation context and should be treated as a hard foundation blocker.

