"""Tests for the Vision Engine adapter."""

import json

import numpy as np
from vision_engine.types import Frame, VisionEvent

from app.adapters.vision import VisionAdapter
from app.config import Settings


class FakeCamera:
    def __init__(self) -> None:
        self._closed = False

    def read(self) -> Frame:
        return Frame(data=np.zeros((480, 640, 3), dtype=np.uint8), timestamp_ns=123)

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


class FakePipeline:
    def process_frame(self, frame: Frame) -> VisionEvent:
        return VisionEvent(
            frame_id=1,
            timestamp_ns=frame.timestamp_ns,
            source="synthetic",
            detections=[],
            meshes=[],
        )


def _adapter() -> VisionAdapter:
    return VisionAdapter(
        Settings(),
        camera=FakeCamera(),
        pipeline=FakePipeline(),
    )


def test_status_shape() -> None:
    status = _adapter().status()
    assert set(status) == {"camera", "pipeline", "error"}
    camera = status["camera"]
    assert isinstance(camera, dict)
    assert set(camera) == {"source", "running", "resolution", "fps"}
    assert camera["source"] == "synthetic"
    assert camera["running"] is True
    assert camera["resolution"] == {"width": 640, "height": 480}
    assert camera["fps"] == 30.0
    assert status["pipeline"] == "stopped"
    assert status["error"] is None


def test_capture_once_returns_json_clean_dict() -> None:
    event = _adapter().capture_once()
    assert set(event) == {
        "frame_id",
        "timestamp_ns",
        "source",
        "face_count",
        "detections",
        "meshes",
    }
    assert event["face_count"] == 0
    payload = json.dumps(event)
    assert isinstance(payload, str)


def test_close_is_idempotent() -> None:
    camera = FakeCamera()
    adapter = VisionAdapter(Settings(), camera=camera, pipeline=FakePipeline())
    adapter.close()
    adapter.close()
    assert camera.closed is True
    status = adapter.status()
    camera_status = status["camera"]
    assert isinstance(camera_status, dict)
    assert camera_status["running"] is False


def test_close_drops_resources_and_next_use_rebuilds_prototypes() -> None:
    camera = FakeCamera()
    adapter = VisionAdapter(Settings(), camera=camera, pipeline=FakePipeline())
    adapter.close()
    assert camera.closed is True

    # A later capture rebuilds from the injected prototypes: the same adapter
    # instance can run again after stop (the manager release/restart cycle).
    event = adapter.capture_once()
    assert isinstance(event, dict)
    assert event["frame_id"] == 1