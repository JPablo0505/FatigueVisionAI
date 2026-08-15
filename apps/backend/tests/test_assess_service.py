"""Service-level tests: VisionEvent -> FeatureExtractor -> AI engine.

These run the real ``FeatureExtractor`` with synthetic VisionEvents, so the
full biometric path (EAR/MAR/PERCLOS/window aggregation) is exercised
without any MediaPipe dependency.
"""

from __future__ import annotations

import numpy as np
from ai_engine import StubFatiguePredictor
from ai_engine.ports import FatigueModel
from ai_engine.types import FatigueFeatures, FatigueResult
from feature_extractor import FeatureExtractor
from vision_engine.types import Frame, VisionEvent

from app.adapters.ai import AIAdapter
from app.adapters.vision import VisionAdapter
from app.config import Settings
from app.services.assessment import NO_FACE_MESSAGE, AssessmentService
from tests.synthetic_faces import build_face_event, build_no_face_event


class FakeCamera:
    def __init__(self) -> None:
        self._closed = False

    def read(self) -> Frame:
        return Frame(data=np.zeros((480, 640, 3), dtype=np.uint8), timestamp_ns=0)

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


class FakePipeline:
    def process_frame(self, frame: Frame) -> VisionEvent:
        return VisionEvent(
            frame_id=0,
            timestamp_ns=frame.timestamp_ns,
            source="synthetic",
            detections=[],
            meshes=[],
        )


class ExplodingPredictor:
    """Fails loudly when called; proves the predictor is untouched on no-face."""

    def predict(self, features: FatigueFeatures) -> FatigueResult:
        raise AssertionError("predictor must not run without features")


def _service(predictor: FatigueModel | None = None) -> AssessmentService:
    vision = VisionAdapter(
        Settings(),
        camera=FakeCamera(),
        pipeline=FakePipeline(),
    )
    ai = AIAdapter(predictor or StubFatiguePredictor())
    return AssessmentService(vision, ai, FeatureExtractor())


def test_assess_event_with_face_forwards_validated_features() -> None:
    service = _service()
    response = service.assess_event(build_face_event(frame_id=1, timestamp_ns=0, ear=0.5))

    assert response.status == "ok"
    assert response.message is None
    assert response.features is not None
    assert response.features.ear == 0.5
    assert response.features.perclos == 0.0
    assert response.assessment is not None
    assert response.assessment.score == 0.0
    assert response.assessment.level == "low"


def test_assess_event_without_face_returns_no_face_and_skips_predictor() -> None:
    service = _service(ExplodingPredictor())
    response = service.assess_event(build_no_face_event(frame_id=1, timestamp_ns=0))
    assert response.status == "no_face"
    assert response.features is None
    assert response.assessment is None
    assert response.message == NO_FACE_MESSAGE


def test_extractor_singleton_accumulates_window_between_calls() -> None:
    service = _service()
    first = service.assess_event(build_face_event(frame_id=1, timestamp_ns=0, ear=0.5))
    assert first.status == "ok"
    assert first.features is not None
    assert first.features.perclos == 0.0

    # A second call must join the SAME window: 1 closed frame out of 2
    # (ear=0.05 < 0.25 closes the eyes). A fresh per-request extractor would
    # report perclos == 1.0 here.
    second = service.assess_event(build_face_event(frame_id=2, timestamp_ns=1, ear=0.05))
    assert second.status == "ok"
    assert second.features is not None
    assert second.features.perclos == 0.5


def test_no_face_frames_advance_window_but_do_not_break_aggregation() -> None:
    service = _service()
    service.assess_event(build_face_event(frame_id=1, timestamp_ns=0, ear=0.5))
    gap = service.assess_event(build_no_face_event(frame_id=2, timestamp_ns=1))
    assert gap.status == "no_face"
    third = service.assess_event(build_face_event(frame_id=3, timestamp_ns=2, ear=0.5))
    assert third.status == "ok"
    assert third.features is not None
    # The no-face frame keeps the window open but contributes no measurements.
    assert third.features.perclos == 0.0
