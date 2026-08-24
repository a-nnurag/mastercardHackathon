"""
Risk Engine & Fusion Layer — Core security decision authority.

Invariants:
- RiskEngine is the SOLE authority producing RiskDecision.
- RiskEngine consumes SignalSet — it does NOT execute detectors or orchestrate training.
- Hard security rule violations cannot result in ALLOW.
- The LLM can explain this decision, but can never create, modify, lower, or override it.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from defend.contracts import RiskDecision, SignalResult, SignalSet


@dataclass
class RiskFusionPolicy:
    high_threshold: float = 0.50
    medium_threshold: float = 0.15
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "constraint_drift": 0.25,
            "lightgbm_prob": 0.25,
            "gnn_prob": 0.20,
            "content_injection": 0.15,
            "utterance_artifact_divergence": 0.10,
            "ingestion_source_trust_score": 0.05,
        }
    )
    hard_violation_action: Literal["BLOCK", "HOLD"] = "BLOCK"
    empty_signals_action: Literal["HOLD", "ALLOW"] = "HOLD"


class RiskEngine:
    def __init__(self, policy: Optional[RiskFusionPolicy] = None):
        self.policy = policy or RiskFusionPolicy()

    def evaluate(
        self,
        signals: SignalSet,
        policy: Optional[RiskFusionPolicy] = None,
        attack_family: Optional[str] = None,
    ) -> RiskDecision:
        p = policy or self.policy

        # Collect evidence and available signals
        evidence = signals.all_evidence()
        signals_dict: dict[str, Any] = {}

        for name, sig in signals.signals.items():
            if sig.available:
                signals_dict[name] = sig.value

        # 1. Hard Security Violation check (Authority rule: hard breach forces non-ALLOW)
        if signals.has_hard_violation():
            decision = p.hard_violation_action
            risk_level = "HIGH"
            fused_score = max(self._compute_fused_score(signals, p), 0.90)

            if attack_family is None:
                attack_family = self._determine_attack_family(signals)

            return RiskDecision(
                risk_score=min(1.0, max(0.0, fused_score)),
                risk_level=risk_level,
                decision=decision,
                signals=signals_dict,
                evidence=evidence or ["hard_security_constraint_breach"],
                attack_family=attack_family,
            )

        # 2. No available signals check — Fail-Closed security principle
        available_numeric_signals = {
            k: v for k, v in signals_dict.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if not available_numeric_signals and not signals_dict:
            decision = p.empty_signals_action
            risk_level = "MEDIUM" if decision == "HOLD" else "LOW"
            return RiskDecision(
                risk_score=0.5 if decision == "HOLD" else 0.0,
                risk_level=risk_level,
                decision=decision,
                signals={},
                evidence=["no_signals_available_hold_for_manual_review"],
                attack_family=attack_family,
            )

        # 3. Pure Weighted Risk Fusion
        fused_score = self._compute_fused_score(signals, p)

        # 4. Threshold-based policy decision
        if fused_score >= p.high_threshold:
            risk_level = "HIGH"
            decision = "BLOCK"
        elif fused_score >= p.medium_threshold:
            risk_level = "MEDIUM"
            decision = "HOLD"
        else:
            risk_level = "LOW"
            decision = "ALLOW"

        if attack_family is None:
            attack_family = self._determine_attack_family(signals)

        return RiskDecision(
            risk_score=min(1.0, max(0.0, fused_score)),
            risk_level=risk_level,
            decision=decision,
            signals=signals_dict,
            evidence=evidence,
            attack_family=attack_family,
        )

    def _compute_fused_score(self, signals: SignalSet, policy: RiskFusionPolicy) -> float:
        total_weight = 0.0
        weighted_sum = 0.0

        for name, weight in policy.weights.items():
            sig = signals.get(name)
            if sig is not None and sig.available and isinstance(sig.value, (int, float)) and not isinstance(sig.value, bool):
                weighted_sum += weight * float(sig.value)
                total_weight += weight

        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight

    def _determine_attack_family(self, signals: SignalSet) -> Optional[str]:
        gnn_prob = signals.get_value("gnn_prob", None)
        if gnn_prob is not None and float(gnn_prob) >= 0.4:
            return "merchant_laundering"
        if (
            signals.get_value("constraint_drift", 0.0) > 0
            or signals.get_value("content_injection", 0.0) >= 0.15
            or signals.get_value("lightgbm_prob", 0.0) >= 0.4
        ):
            return "prompt_injection"
        return None
