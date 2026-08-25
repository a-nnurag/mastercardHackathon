# Attack Surface Rulebook: Mastercard GenAI Fraud Defense Platform

This permanent rulebook defines how to design, implement, test, and register a **new attack family/surface** in the Mastercard GenAI Fraud Defense codebase without breaking shared contracts or core architectural boundaries.

---

## 1. Architectural Model & Purpose

The defense platform is organized into three core pillars connected by a closed-loop mutation cycle:

```
[ IDENTIFY ]   Threat Intelligence / Attack Taxonomy (identify/)
     ↓
[ GENERATE ]   Synthetic Simulation & Narrative Generation (generate/)
     ↓
[ DEFEND ]     Multi-Layer Detection Pipeline & Risk Engine (defend/)
     ↓
[ DECISION ]   Canonical RiskDecision (ALLOW / HOLD / BLOCK)
     ↓
[ EXPLAIN ]    Non-Authoritative LLM Analyst Narrative (defend/llm_verdict.py)
     ↓
[ MUTATE ]     Confidence-Guided Mutation Feedback Loop (mutator/)
```

### Why Attack Families Are Plugins
Fraud schemes in Agentic Commerce (indirect prompt injection, shell merchant networks, deepfake biometrics, autonomous agent impersonation, synthetic merchant storefronts) evolve rapidly. Treating attack families as modular plugins allows teammates to work in parallel on separate attack vectors without modifying the core risk engine or breaking shared execution paths.

---

## 2. Core vs. Plugin Boundaries

| Layer | Owned by Core | Owned by Plugin / Attack Family |
|---|---|---|
| **Pillar: Identify** | Taxonomy schema (`AttackVector`), JSON store (`taxonomy.json`), RAG indexing loop. | Attack pattern definitions, taxonomy groundings, threat report sources. |
| **Pillar: Generate** | Session schema (`AgentSession`), mandate structure (`MandateScope`), IEEE-CIS dataset joins. | Attack scenario generation, red-team prompt templates, domain generation. |
| **Pillar: Defend** | `DetectionPipeline`, `SignalSet`, `DetectionContext`, `RiskEngine`, `ModelRegistry`, `RiskDecision`. | Attack-specific `SignalDetector` classes producing standard `SignalResult` outputs. |
| **Pillar: Explain** | `synthesize_explanation()`, `RiskExplanation` contract, analyst templates. | Attack-specific explanation prompt context and evidence bullets. |
| **Pillar: Mutate** | Cross-validation runner, hard-session mining (`find_hard_sessions`), round cache. | Mutation prompts and variant generation heuristics. |

### Strict Non-Negotiable Core Invariants
A plugin must **NEVER**:
1. ❌ Directly emit a final `ALLOW` / `HOLD` / `BLOCK` decision (Only `RiskEngine` is the decision authority).
2. ❌ Mutate the `AgentSession` dataclass dynamically at runtime.
3. ❌ Train or fine-tune models inside runtime inference paths.
4. ❌ Inspect privileged ground-truth labels (e.g. `session.injection_present`, `session.injection_payload_text`) inside blue-team detectors.
5. ❌ Introduce custom session schemas that bypass `AgentSession`.

---

## 3. Standard Attack Lifecycle

Every attack family implementation follows this 7-step lifecycle:

```
1. ATTACK DEFINITION  ──> Define entry in identify/taxonomy.json
         ↓
2. GENERATION         ──> Implement red-team generator producing AgentSession instances
         ↓
3. SIMULATION         ──> Model environment / network artifacts (e.g. Graph, Cart, API calls)
         ↓
4. DETECTION          ──> Implement SignalDetector returning SignalResult
         ↓
5. RISK FUSION        ──> Register SignalDetector into DetectionPipeline & RiskEngine
         ↓
6. FAILURE MINING     ──> Identify false negatives / low-confidence detections
         ↓
7. MUTATION & RETEST  ──> Mutate evasive variants and verify defense hardening
```

---

## 4. Canonical Data Contracts

### 4.1 Input Session Contract (`generate/session_schema.py`)
All session data passed to detectors must conform to `AgentSession`:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class MandateScope:
    categories: list[str]
    amount_cap: float
    merchant_allowlist: list[str]
    expiry: Optional[object] = None

@dataclass
class AgentSession:
    agent_id: str
    agent_registry_status: str           # "valid", "suspended", "unregistered"
    mandate_scope: MandateScope
    intent_artifact_hash: str
    raw_utterance: str                  # Human prompt
    signed_artifact_text: str          # What the agent cryptographically signed
    task_origin_url: str               # Origin URL / domain
    content_sources_ingested: list[str]# Domains browsed
    hops_since_intent: int = 0
    tool_calls_made: int = 0
    injection_present: bool = False     # Ground truth (Red-team ONLY, never blue-team)
    injection_payload_text: Optional[str] = None # Ground truth payload
```

### 4.2 Detection Signal Contract (`defend/contracts.py`)
Every detector must return a `SignalResult`:

```python
@dataclass
class SignalResult:
    name: str                           # Unique signal identifier (e.g. "gnn_prob", "content_injection")
    value: Any                          # Float [0.0, 1.0], bool, or int
    available: bool = True              # False if data/model unavailable
    evidence: list[str] = field(...)    # Human-readable audit findings
    hard_violation: bool = False        # True IF hard policy rule breached
    metadata: dict[str, Any] = field(...)
```

---

## 5. Implementing a New Attack Family: Step-by-Step

### Step 1: Add Threat Taxonomy Entry (`identify/`)
Add your attack pattern to `identify/taxonomy.json` using the `AttackVector` schema:
```json
{
  "attack_id": "ATK-03-DEEPFAKE-KYC",
  "name": "Synthetic Biometric Injection",
  "delivery": "api_gateway_injection",
  "mechanism": "Generative adversarial synthesis bypassing biometric verification",
  "target": "agent_identity_registry",
  "grounding_sources": ["03_csa_ap2_stride_maestro.txt"]
}
```

### Step 2: Implement Red-Team Generator (`generate/`)
Generate paired sessions with both obvious and subtle variants:
```python
def generate_attack_sessions(count: int, seed: int = 42) -> list[AgentSession]:
    # Produce reproducible AgentSession instances with ground-truth injection_present=True/False
    ...
```

### Step 3: Implement Blue-Team `SignalDetector` (`defend/`)
Create a class inheriting from `SignalDetector`:
```python
from defend.contracts import SignalResult
from defend.detection_context import DetectionContext
from defend.signals.base import SignalDetector
from generate.session_schema import AgentSession

class DeepfakeKYCDetector(SignalDetector):
    name: str = "deepfake_kyc_score"
    supported_attack_families: tuple[str, ...] = ("biometric_spoofing",)

    def detect(self, session: AgentSession, context: Optional[DetectionContext] = None) -> SignalResult:
        # 1. Read ONLY visible session facts and context metadata
        # 2. Compute anomaly / classification score in [0.0, 1.0]
        score = ... 
        evidence = [f"biometric_anomaly_detected: score {score:.3f}"] if score >= 0.5 else []

        return SignalResult(
            name=self.name,
            value=score,
            available=True,
            evidence=evidence,
            hard_violation=False,
            metadata={"raw_score": score},
        )
```

### Step 4: Register Detector in `DetectionPipeline`
Register your detector in `defend/pipeline.py`:
```python
def get_default_detectors() -> list[SignalDetector]:
    return [
        RuleDetector(),
        ConstraintDriftDetector(),
        IngestionSourceTrustDetector(),
        ContentInjectionDetector(),
        DivergenceDetector(),
        LightGBMDetector(),
        GNNDetector(),
        DeepfakeKYCDetector(), # Added new detector
    ]
```

### Step 5: Configure `RiskFusionPolicy` Weights (`defend/risk_engine.py`)
Add your signal to the fusion policy weights:
```python
RiskFusionPolicy(
    weights={
        "constraint_drift": 0.20,
        "lightgbm_prob": 0.20,
        "gnn_prob": 0.20,
        "deepfake_kyc_score": 0.15, # Signal weight
        "content_injection": 0.10,
        "utterance_artifact_divergence": 0.10,
        "ingestion_source_trust_score": 0.05,
    }
)
```

---

## 6. Testing & Quality Requirements

Every new attack plugin must provide a dedicated test file in `tests/test_<attack_name>.py` covering:

1. **Contract Validity**: Asserts detector returns valid `SignalResult` bounded in $[0.0, 1.0]$.
2. **Obvious Attack Separation**: Asserts clear attack scenarios receive high risk scores ($\ge 0.50$).
3. **Clean Session Non-Interference**: Asserts clean benign sessions do not trigger false positive breaches.
4. **Missing Data Resilience**: Asserts detector gracefully returns `available=False` when upstream data/models are missing.
5. **No Model Training in Inference**: Asserts calling `detect()` does not alter model weights or perform gradient updates.

---

## 7. Definition of Done Checklist

A new attack surface PR is ready for merge only when:
- [ ] Added taxonomy pattern in `identify/taxonomy.json`.
- [ ] Synthetic generator produces reproducible `AgentSession` pairs with fixed seeds.
- [ ] Detector implements `SignalDetector(ABC)` with 0 label leakage.
- [ ] Detector registered in `DetectionPipeline` and weighted in `RiskFusionPolicy`.
- [ ] Offline training (if ML model used) lives in `defend/training/` and artifacts load via `ModelRegistry`.
- [ ] Unit tests pass with `pytest tests/test_<attack>.py -v`.
- [ ] Entire repository test suite passes with `pytest -v` (100% green).
