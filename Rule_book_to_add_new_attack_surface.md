Prompt: Create the Attack Surface Rulebook for the Mastercard GenAI Fraud Defense Platform

You are working on the Mastercard Innovation Challenge 2026 project.

The platform follows this architecture:

Threat Intelligence / Attack Taxonomy ↓ RED-TEAM ENGINE generate /
simulate / mutate ↓ BLUE-TEAM ENGINE identity / transaction / network
signals ↓ RISK FUSION ↓ ALLOW / HOLD / BLOCK ↓ EXPLANATION + AUDIT ↓
FEEDBACK LOOP

The immediate foundation commit is being stabilized first. After that,
multiple developers will work in parallel on different attack families.

Your task is to create a permanent repository-level document:

docs/ATTACK_SURFACE_RULEBOOK.md

This rulebook must allow a new developer to implement a completely new
GenAI fraud attack family without modifying the core architecture
unnecessarily.

What the rulebook must define

1. Purpose

Explain: - what an attack surface/plugin is; - why attack families are
plugins; - what the core engine owns; - what a plugin owns.

2. Core abstraction

Define a language/framework-neutral conceptual interface such as:

AttackSurface
├── metadata()
├── generate()
├── simulate()
├── detect()
├── evaluate()
├── mutate()
└── cleanup/reset if required

Adapt the exact method names to the existing codebase rather than
blindly inventing them.

For every method specify: - input contract; - output contract; - side
effects; - error behavior; - deterministic/reproducibility
expectations; - what must NOT be done.

3. Standard attack lifecycle

Every plugin must follow:

ATTACK DEFINITION
      ↓
GENERATION
      ↓
SIMULATION
      ↓
DETECTION
      ↓
EVALUATION
      ↓
FAILURE MINING
      ↓
MUTATION
      ↓
RETEST

Explain what happens at each stage.

4. Common data contracts

Define schemas/contracts for: - attack scenario; - simulated
session/event; - attack evidence; - detection signal; - attack result; -
mutation result; - evaluation metrics.

Do not hardcode fields that only belong to one attack family.

Use extensible metadata/configuration where attack-specific fields are
needed.

5. Core/plugin boundary

Explicitly document:

Core owns: - orchestration; - event/session lifecycle; - risk fusion; -
final decision; - common telemetry; - common evaluation; -
audit/logging; - plugin discovery/registration; - common
configuration; - security boundaries.

Plugin owns: - attack-specific generation; - attack simulation; -
attack-specific features/signals; - attack-specific mutations; -
attack-specific evaluation logic where necessary.

A plugin must NOT: - directly change the final risk decision; - bypass
the risk engine; - modify another plugin; - secretly train production
models during inference; - introduce a second incompatible session
schema; - duplicate common infrastructure unnecessarily.

6. Detection contract

A plugin should return structured evidence/signals, not directly return
"ALLOW/BLOCK" as the final authority.

Example:

{
  "attack_family": "example",
  "risk_score": 0.87,
  "signals": {
    "signal_name": 0.91
  },
  "evidence": [
    {
      "type": "example",
      "description": "...",
      "severity": "high"
    }
  ],
  "confidence": 0.92
}

The exact schema must match the actual stabilized codebase.

7. Red-team requirements

Every attack plugin must: - generate realistic scenarios; - support
reproducible seeds; - include both obvious and subtle variants; -
document assumptions; - avoid depending on inaccessible private
Mastercard data; - clearly separate synthetic assumptions from
real-world facts; - provide enough metadata to reproduce an attack.

8. Blue-team requirements

Every plugin must define: - what signals it expects; - what evidence it
produces; - how false positives/false negatives are measured; - what
constitutes successful defense; - what happens when required signals are
unavailable.

Do not force every attack family to use the same ML model.

9. Mutation requirements

Mutation should make attacks harder without simply producing random
noise.

Mutations should target meaningful attack dimensions such as: -
wording; - timing; - sequence; - amount; - entity relationships; -
source/provenance; - multi-step behavior; - evasion strategy.

Every mutation should retain: - parent attack ID; - mutation type; -
seed; - parameters; - resulting scenario; - detector result.

10. Evaluation requirements

Every plugin must report at least: - attack generation count; - attack
success rate; - detection rate/recall; - false-positive impact where
measurable; - latency; - mutation effectiveness; - performance on
unseen/subtle variants.

Prefer PR-AUC/recall/FPR and business-impact metrics where applicable
rather than relying only on ROC-AUC.

11. Testing requirements

Every plugin must contain: - unit tests; - schema/contract tests; -
deterministic generation tests; - happy-path attack tests; -
subtle/evasion attack tests; - regression tests for previously missed
attacks.

Plugins must not break core tests.

12. Registration/discovery

Document exactly how a developer: 1. creates a plugin directory; 2.
implements the abstraction; 3. adds configuration; 4.
registers/discovers the plugin; 5. runs its tests; 6. runs it through
the common evaluation pipeline.

Prefer automatic discovery/registration if it fits the existing
codebase, but do not introduce unnecessary complexity.

13. Git/parallel-development rules

Each attack family should normally live in its own directory/module.

Example:

attack_surfaces/
├── agent_hijacking/
├── shell_merchant/
├── synthetic_identity/
├── deepfake_kyc/
├── fake_storefront/
└── ai_scams/

Developers should: - work in separate branches; - avoid modifying core
files unless the abstraction is genuinely insufficient; - add tests with
the plugin; - document any required core change; - keep commits
focused; - never silently change shared contracts.

14. Definition of Done

A new attack surface is complete only when: - it implements the common
abstraction; - it can generate scenarios; - it can simulate them; - it
returns standardized evidence/signals; - it participates in common risk
evaluation; - it supports mutation/evolution; - it has tests; - it has
reproducible evaluation; - it is documented; - it can be
enabled/disabled without breaking other attack surfaces.

Important instruction

Before writing the rulebook, inspect the actual stabilized codebase and
adapt the abstraction to the real architecture.

Do NOT invent an abstract framework disconnected from the repository.

Do NOT rewrite working core architecture merely to make the
documentation cleaner.

The rulebook should preserve the project's current design while
establishing a stable extension contract for future attack families.

The final document should be practical enough that a teammate can read
it and immediately implement Attack 3, Attack 4, Attack 5, or Attack 6
independently.