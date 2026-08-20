# Integration contract — Task 12

Implementation: [`defend/integration_contract.py`](defend/integration_contract.py).
Tests: [`tests/test_integration_contract.py`](tests/test_integration_contract.py) (5/5 passing).

## Where this runs

Per `TEAM_BRIEF.md` §3.3: `constraint_drift`/`utterance_artifact_divergence`
need the **raw human utterance**, which Mastercard's network never sees.
So this cannot run at the network layer. It runs at the **agent-provider
or PSP layer**, where the utterance lives, and emits a **session-risk
score that rides along with the signed Intent Artifact** into the
network. Mastercard's Decision Intelligence **consumes** that score — it
does not compute it (`TEAM_BRIEF.md` Part 9 Q&A). No new network
infrastructure is required.

## What's in the payload, and what's deliberately not

The payload is built from the four fast, deterministic/ML Defend layers
only — rules (Task 3), LightGBM (Task 5), content similarity (Task 7),
GNN (Task 8). Task 9's LLM narrative verdict (`defend/llm_verdict.py`) is
**intentionally excluded** from this path: it's a slower, on-demand
explanation layer for a human risk analyst (see the dashboard's
"Get AI Verdict" button, Task 11), not something that belongs in a
per-transaction latency budget or in front of a machine consumer that
only needs a number to threshold on.

`session_risk_score` is `max(lightgbm_prob, gnn_prob)` — the higher of
the two attack-type-specific supervised scores, since either attack type
(#1 prompt injection, #2 merchant laundering) should be enough to raise
risk. `risk_level` is a deterministic function of that score plus the
hard rules check:

```
HIGH   if rules_flagged OR score >= 0.5
MEDIUM if score >= 0.15 OR content_score >= 0.15
LOW    otherwise
```

The 0.5/0.15 thresholds match the ones already used in
`defend/llm_verdict.py`'s `consistency_check()`, kept consistent across
both consumers of these same four signals rather than inventing a second
set of cutoffs.

## JSON Schema

```json
{
  "type": "object",
  "properties": {
    "schema_version": { "type": "string" },
    "agent_id": { "type": "string" },
    "intent_artifact_hash": {
      "type": "string",
      "description": "Ties this score to the specific signed Intent Artifact it rides along with."
    },
    "session_risk_score": {
      "type": "number", "minimum": 0.0, "maximum": 1.0,
      "description": "max(lightgbm_prob, gnn_prob) -- the single number a Decision Intelligence-style consumer thresholds on."
    },
    "risk_level": { "type": "string", "enum": ["LOW", "MEDIUM", "HIGH"] },
    "contributing_signals": {
      "type": "object",
      "properties": {
        "rules_flagged": { "type": "boolean" },
        "rules_reasons": { "type": "array", "items": { "type": "string" } },
        "lightgbm_prob": { "type": "number" },
        "content_score": { "type": "number" },
        "gnn_prob": { "type": ["number", "null"] }
      },
      "required": ["rules_flagged", "rules_reasons", "lightgbm_prob", "content_score", "gnn_prob"]
    }
  },
  "required": ["schema_version", "agent_id", "intent_artifact_hash", "session_risk_score", "risk_level", "contributing_signals"]
}
```

(The live, importable copy is `defend.integration_contract.SESSION_RISK_PAYLOAD_SCHEMA`
— validated in CI via `jsonschema.validate()` in `test_build_session_risk_payload_matches_json_schema`,
not just documented here as prose.)

## Real example payloads

Both pulled from a live run against the real 226-session dataset — not hand-written.

**A legitimate booking** (`agent-cfb4637f`, clean travel session):

```json
{
  "schema_version": "1.0",
  "agent_id": "agent-cfb4637f",
  "intent_artifact_hash": "b70c346943cacee4",
  "session_risk_score": 0.2057,
  "risk_level": "MEDIUM",
  "contributing_signals": {
    "rules_flagged": false,
    "rules_reasons": [],
    "lightgbm_prob": 0.0001,
    "content_score": 0.0,
    "gnn_prob": 0.2057
  }
}
```

**Honest caveat, not smoothed over**: this benign session lands on
`MEDIUM`, not `LOW`, purely because of `gnn_prob=0.2057` — the GNN's raw
probability for a merchant that isn't a training-set fraud-ring member
but also isn't one of the 19 hand-labeled legitimate merchants doesn't
reliably sit near zero. This is a real limitation of Task 8's GNN as
currently trained (66-node held-out test set, `defend/gnn.py`), not a
bug in this contract layer — flagging it here rather than picking a
cherry-picked example that hides it.

**A prompt-injection hijack** (`agent-3612bb96`, budget travel booking
hijacked into an unrelated high-value camera purchase at a typosquat
domain):

```json
{
  "schema_version": "1.0",
  "agent_id": "agent-3612bb96",
  "intent_artifact_hash": "ecad96e6f3ffc191",
  "session_risk_score": 1.0,
  "risk_level": "HIGH",
  "contributing_signals": {
    "rules_flagged": true,
    "rules_reasons": [
      "amount cap breached: signed INR 350,000 > cap INR 8,000",
      "task_origin_url domain 'makemytrip-offer.com' not in merchant allowlist ['cleartrip.com', 'makemytrip.com']"
    ],
    "lightgbm_prob": 1.0,
    "content_score": 0.0345,
    "gnn_prob": 0.989
  }
}
```

## Real measured latency

Measured by `defend.integration_contract.measure_latency()`: one
LightGBM model fit once and held in memory (the realistic
production shape — train offline, serve from memory, never refit
per request) and the GNN's merchant-probability lookup built once,
then 50 single-session payloads timed individually end-to-end
(rules check + one LightGBM `predict()` call + content-similarity
scoring + one dict lookup), sampled across the real 226-session
dataset, first call excluded as warm-up:

| Metric | Value |
|---|---|
| n runs | 50 |
| mean | **9.08 ms** |
| p50 | 8.88 ms |
| p95 | 10.59 ms |
| max | 10.74 ms |

Reproduce: `python3 -m defend.integration_contract` (prints the latency
table and both example payloads above, freshly computed).
