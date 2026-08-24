"""
Base Signal Detector abstraction.

All signal detectors (heuristics, NLP, tabular models, graph networks) implement
this common interface, returning a standard SignalResult.
"""

from abc import ABC, abstractmethod
from typing import Optional

from defend.contracts import SignalResult
from defend.detection_context import DetectionContext
from generate.session_schema import AgentSession


class SignalDetector(ABC):
    name: str = "base_detector"
    supported_attack_families: tuple[str, ...] = ("prompt_injection",)

    @abstractmethod
    def detect(
        self,
        session: AgentSession,
        context: Optional[DetectionContext] = None,
    ) -> SignalResult:
        """
        Inspect an AgentSession (and optional DetectionContext) and emit a SignalResult.
        """
        raise NotImplementedError
