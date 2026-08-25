"""
Detection Pipeline Orchestrator — Executes SignalDetectors on AgentSession + DetectionContext
to produce a canonical SignalSet for RiskEngine evaluation.
"""

from typing import Iterable, Optional

from defend.contracts import SignalResult, SignalSet
from defend.detection_context import DetectionContext
from defend.signals.base import SignalDetector
from defend.constraint_drift import ConstraintDriftDetector, IngestionSourceTrustDetector
from defend.rules import RuleDetector
from defend.content_layer import ContentInjectionDetector
from defend.divergence import DivergenceDetector
from defend.lightgbm_baseline import LightGBMDetector
from defend.gnn import GNNDetector
from generate.session_schema import AgentSession


def get_default_detectors() -> list[SignalDetector]:
    """Instantiates the standard 7-detector defense suite."""
    return [
        RuleDetector(),
        ConstraintDriftDetector(),
        IngestionSourceTrustDetector(),
        ContentInjectionDetector(),
        DivergenceDetector(),
        LightGBMDetector(),
        GNNDetector(),
    ]


class DetectionPipeline:
    """
    Standard orchestrator that executes registered SignalDetectors over an AgentSession.
    """

    def __init__(
        self,
        detectors: Optional[Iterable[SignalDetector]] = None,
        context: Optional[DetectionContext] = None,
    ):
        self.detectors: list[SignalDetector] = list(detectors) if detectors is not None else get_default_detectors()
        self.context: Optional[DetectionContext] = context

    def add_detector(self, detector: SignalDetector) -> None:
        self.detectors.append(detector)

    def run(
        self,
        session: AgentSession,
        context: Optional[DetectionContext] = None,
    ) -> SignalSet:
        ctx = context or self.context
        signal_set = SignalSet()

        for detector in self.detectors:
            try:
                res = detector.detect(session, ctx)
                signal_set.add(res)
            except Exception as e:
                # Resilient execution: record failed detector as unavailable with error evidence
                signal_set.add(
                    SignalResult(
                        name=getattr(detector, "name", detector.__class__.__name__),
                        value=0.0,
                        available=False,
                        evidence=[f"detector_error: {type(e).__name__}: {e}"],
                    )
                )

        return signal_set
