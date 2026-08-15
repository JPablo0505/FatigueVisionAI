"""Tests for /api/healthz and /api/vision endpoints."""

import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from vision_engine.types import Frame, VisionEvent

from app.adapters.vision import VisionAdapter
from app.config import Settings
from app.di import create_app


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


def _client() -> TestClient:
    vision = VisionAdapter(
        Settings(),
        camera=FakeCamera(),
        pipeline=FakePipeline(),
    )
    return TestClient(create_app(vision=vision))


def test_healthz() -> None:
    response = _client().get("/api/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_vision_status() -> None:
    response = _client().get("/api/vision/status")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"camera", "pipeline", "error"}
    assert body["camera"]["source"] == "synthetic"
    assert body["pipeline"] == "stopped"


def test_vision_frame() -> None:
    response = _client().post("/api/vision/frame")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "frame_id",
        "timestamp_ns",
        "source",
        "face_count",
        "detections",
        "meshes",
    }
    assert body["face_count"] == 0


def _models_present() -> bool:
    settings = Settings()
    return settings.face_detector_model.is_file() and settings.face_landmarker_model.is_file()


def _integration_enabled() -> bool:
    return os.getenv("FATIGUE_INTEGRATION_TESTS") == "1"


@pytest.mark.skipif(
    not _models_present() or not _integration_enabled(),
    reason="HTTP-level MediaPipe integration is opt-in: needs models + FATIGUE_INTEGRATION_TESTS=1"
    " (on Windows, mediapipe workers + TestClient portal hang the interpreter at exit)",
)
def test_vision_frame_real_pipeline() -> None:
    client = TestClient(create_app())
    response = client.post("/api/vision/frame")
    assert response.status_code == 200
    assert response.json()["face_count"] >= 0


@pytest.mark.skipif(
    not _models_present(),
    reason="MediaPipe models not downloaded; requires network/model files",
)
def test_real_pipeline_capture_once_direct() -> None:
    adapter = VisionAdapter(Settings())
    event = adapter.capture_once()
    assert isinstance(event, dict)
    face_count = event["face_count"]
    assert isinstance(face_count, int)
    assert face_count >= 0