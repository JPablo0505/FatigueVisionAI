"""Unit tests for VisionPipeline using deterministic fakes (no hardware)."""

import json
from typing import Any

import numpy as np

from vision_engine import (
    FaceDetection,
    FaceLandmarks,
    Frame,
    VisionPipeline,
)
from vision_engine.ports import FaceDetector, FaceMeshEstimator, FrameProcessor


class StubDetector(FaceDetector):
    """FaceDetector fake returning a fixed detection."""

    def detect(self, frame: Frame) -> list[FaceDetection]:
        return [FaceDetection(bbox=(1, 2, 30, 40), score=0.95)]


class StubMesh(FaceMeshEstimator):
    """FaceMeshEstimator fake returning a fixed landmark mesh."""

    def estimate(self, frame: Frame) -> list[FaceLandmarks]:
        return [FaceLandmarks(points=((0.5, 0.5, 0.0),))]


class CapturingDetector(FaceDetector):
    """FaceDetector fake that records the width of the frame it received."""

    def __init__(self) -> None:
        self.last_width: int | None = None

    def detect(self, frame: Frame) -> list[FaceDetection]:
        self.last_width = frame.width
        return []


class UpscaledProcessor(FrameProcessor):
    """FrameProcessor fake that doubles the frame width."""

    def process(self, frame: Frame) -> Frame:
        doubled = np.repeat(frame.data, 2, axis=1)
        return Frame(data=doubled, timestamp_ns=frame.timestamp_ns)


def make_frame() -> Frame:
    """Return a synthetic 3-channel frame with a fixed timestamp."""
    return Frame(np.zeros((240, 320, 3), dtype=np.uint8), timestamp_ns=1_000)


def test_pipeline_without_stages_returns_empty_event() -> None:
    event = VisionPipeline().process_frame(make_frame())
    assert event.frame_id == 1
    assert event.timestamp_ns == 1_000
    assert event.source == "camera"
    assert event.detections == []
    assert event.meshes == []


def test_pipeline_increments_frame_id_across_calls() -> None:
    pipeline = VisionPipeline()
    assert pipeline.process_frame(make_frame()).frame_id == 1
    assert pipeline.process_frame(make_frame()).frame_id == 2


def test_pipeline_collects_detector_and_mesh_outputs() -> None:
    event = VisionPipeline(detector=StubDetector(), mesh=StubMesh()).process_frame(
        make_frame()
    )
    assert len(event.detections) == 1
    assert event.detections[0].bbox == (1, 2, 30, 40)
    assert event.detections[0].score == 0.95
    assert len(event.meshes) == 1
    assert event.meshes[0].points == ((0.5, 0.5, 0.0),)


def test_pipeline_applies_processor_before_other_stages() -> None:
    detector = CapturingDetector()
    pipeline = VisionPipeline(processor=UpscaledProcessor(), detector=detector)
    pipeline.process_frame(make_frame())
    assert detector.last_width == 640


def test_vision_event_to_dict_is_json_clean() -> None:
    event = VisionPipeline(detector=StubDetector(), mesh=StubMesh()).process_frame(
        make_frame()
    )
    payload: dict[str, Any] = event.to_dict()
    json.dumps(payload)
    assert payload["frame_id"] == 1
    assert payload["face_count"] == 1
    assert isinstance(payload["detections"][0]["bbox"], list)


def test_contract_types_serialize_to_plain_javascript_values() -> None:
    detection = FaceDetection(bbox=(1, 2, 30, 40), score=0.95).to_dict()
    mesh = FaceLandmarks(points=((0.25, 0.5, 0.0),)).to_dict()
    assert detection == {"bbox": [1, 2, 30, 40], "score": 0.95}
    assert mesh == {"points": [[0.25, 0.5, 0.0]]}


def test_frame_derives_shape_properties() -> None:
    color = Frame(np.zeros((240, 320, 3), dtype=np.uint8), timestamp_ns=0)
    gray = Frame(np.zeros((240, 320), dtype=np.uint8), timestamp_ns=0)
    assert (color.width, color.height, color.channels) == (320, 240, 3)
    assert (gray.width, gray.height, gray.channels) == (320, 240, 1)