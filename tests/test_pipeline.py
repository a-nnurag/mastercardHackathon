import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defend.contracts import RiskDecision, SignalResult, SignalSet
from defend.detection_context import DetectionContext
from defend.pipeline import DetectionPipeline, get_default_detectors
from defend.risk_engine import RiskEngine
from defend.signals.base import SignalDetector
from generate.session_schema import AgentSession, MandateScope
from generate.synthetic_sessions import LEGITIMATE, HIJACKED


def test_detection_pipeline_runs_all_default_detectors():
    pipeline = DetectionPipeline()
    assert len(pipeline.detectors) == 7

    session = LEGITIMATE[0]
    signal_set = pipeline.run(session)

    assert isinstance(signal_set, SignalSet)
    assert "rules" in signal_set.signals
    assert "constraint_drift" in signal_set.signals
    assert "ingestion_source_trust_score" in signal_set.signals
    assert "content_injection" in signal_set.signals
    assert "utterance_artifact_divergence" in signal_set.signals
    assert "lightgbm_prob" in signal_set.signals
    assert "gnn_prob" in signal_set.signals


def test_detection_pipeline_end_to_end_with_risk_engine():
    pipeline = DetectionPipeline()
    engine = RiskEngine()

    # Clean session
    clean_session = LEGITIMATE[0]
    clean_signals = pipeline.run(clean_session)
    clean_decision = engine.evaluate(clean_signals)

    assert isinstance(clean_decision, RiskDecision)
    assert clean_decision.decision in ("ALLOW", "HOLD")
    assert clean_decision.risk_level in ("LOW", "MEDIUM")

    # Hijacked session
    hijacked_session = HIJACKED[0]
    hijacked_signals = pipeline.run(hijacked_session)
    hijacked_decision = engine.evaluate(hijacked_signals)

    assert isinstance(hijacked_decision, RiskDecision)
    assert hijacked_decision.decision == "BLOCK"
    assert hijacked_decision.risk_level == "HIGH"


class FailingDetector(SignalDetector):
    name: str = "failing_detector"
    def detect(self, session, context=None):
        raise RuntimeError("Simulated hardware/network timeout")


def test_detection_pipeline_resilience_to_detector_failure():
    pipe = DetectionPipeline(detectors=[FailingDetector()])
    session = LEGITIMATE[0]
    signal_set = pipe.run(session)

    res = signal_set.get("failing_detector")
    assert res is not None
    assert res.available is False
    assert any("Simulated hardware/network timeout" in ev for ev in res.evidence)
