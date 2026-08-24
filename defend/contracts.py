"""
Core Defend Security Contracts & Signal Containers.

Invariants:
- SignalResult and SignalSet are the canonical containers for detector outputs.
- DetectionEvidence captures granular evidence strings from individual detectors.
- RiskDecision is the canonical output of RiskEngine.
- RiskExplanation is the strictly non-authoritative output of the LLM explanation layer.
- LLM cannot create, modify, lower, or override RiskDecision.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class SignalResult:
    name: str
    value: float | bool | str | None
    available: bool = True
    evidence: list[str] = field(default_factory=list)
    hard_violation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "available": self.available,
            "evidence": list(self.evidence),
            "hard_violation": self.hard_violation,
            "metadata": dict(self.metadata),
        }


@dataclass
class SignalSet:
    signals: dict[str, SignalResult] = field(default_factory=dict)

    def add(self, signal: SignalResult) -> None:
        self.signals[signal.name] = signal

    def get(self, name: str, default: Optional[SignalResult] = None) -> Optional[SignalResult]:
        return self.signals.get(name, default)

    def get_value(self, name: str, default: Any = None) -> Any:
        sig = self.signals.get(name)
        return sig.value if sig is not None and sig.available else default

    def has_hard_violation(self) -> bool:
        return any(sig.hard_violation for sig in self.signals.values() if sig.available)

    def all_evidence(self) -> list[str]:
        evidence: list[str] = []
        for sig in self.signals.values():
            if sig.available and sig.evidence:
                evidence.extend(sig.evidence)
        return evidence

    def to_dict(self) -> dict[str, Any]:
        return {name: sig.to_dict() for name, sig in self.signals.items()}


@dataclass
class DetectionEvidence:
    signal_name: str
    value: Any
    confidence: Optional[float] = None
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_name": self.signal_name,
            "value": self.value,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "metadata": dict(self.metadata),
        }


@dataclass
class RiskDecision:
    risk_score: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["ALLOW", "HOLD", "BLOCK"]
    signals: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    attack_family: Optional[str] = None

    def __post_init__(self):
        if not (0.0 <= self.risk_score <= 1.0):
            raise ValueError(f"risk_score must be in [0.0, 1.0], got {self.risk_score}")
        if self.risk_level not in ("LOW", "MEDIUM", "HIGH"):
            raise ValueError(f"Invalid risk_level: {self.risk_level}")
        if self.decision not in ("ALLOW", "HOLD", "BLOCK"):
            raise ValueError(f"Invalid decision: {self.decision}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level,
            "decision": self.decision,
            "signals": dict(self.signals),
            "evidence": list(self.evidence),
            "attack_family": self.attack_family,
        }


@dataclass
class RiskExplanation:
    summary: str
    evidence_summary: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "evidence_summary": list(self.evidence_summary),
        }
