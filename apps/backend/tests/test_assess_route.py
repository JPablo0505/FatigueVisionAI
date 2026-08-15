"""HTTP-level tests for the /api/assess endpoint.

The route captures a frame through the configured camera and pipeline; these
tests inject a VisionAdapter with a scripted pipeline that returns synthetic
VisionEvents, so the full flow runs without any MediaPipe dependency.
"""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient
from vision_engine.errors import CameraNotAvailableError, ProcessingError
from vision_engine.types import Frame, VisionEvent

from app.adapters.vision import VisionAdapter
from app.config import Settings
from app.di import create_app
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


class ScriptedPipeline:
    """Returns one event per ``process_frame`` call, in order."""

    def __init__(self, events: list[VisionEvent]) -> None:
        self._events = events

    def process_frame(self, frame: Frame) -> VisionEvent:
        return self._events.pop(0)


def _client(events: list[VisionEvent]) -> TestClient:
    vision = VisionAdapter(
        Settings(),
        camera=FakeCamera(),
        pipeline=ScriptedPipeline(events),
    )
    return TestClient(create_app(vision=vision))


def test_assess_with_face_returns_assessment() -> None:
    client = _client([build_face_event(frame_id=1, timestamp_ns=0, ear=0.5)])
    response = client.post("/api/assess")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["message"] is None
    assert body["features"]["ear"] == 0.5
    assert isinstance(body["assessment"]["timestamp"], float)
    # Production wiring runs the HeuristicFatigueModel (S1-17): a single clean
    # open-eye frame yields the deterministic "normal" state with its base
    # score, not the stub's 0.0.
    assert body["assessment"]["score"] == 16.0
    assert body["assessment"]["level"] == "low"
    assert body["assessment"]["state"] == "normal"


def test_assess_without_face_returns_no_face() -> None:
    client = _client([build_no_face_event(frame_id=1, timestamp_ns=0)])
    response = client.post("/api/assess")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_face"
    assert body["features"] is None
    assert body["assessment"] is None
    assert body["message"] is not None


def test_assess_accumulates_window_across_requests() -> None:
    # Two separate HTTP calls must feed the SAME shared extractor window:
    # the second frame (eyes closed) joins the first one (eyes open), so
    # PERCLOS is 1 closed frame over 2. A per-request extractor would report
    # perclos == 1.0 (single-frame window).
    client = _client(
        [
            build_face_event(frame_id=1, timestamp_ns=0, ear=0.5),
            build_face_event(frame_id=2, timestamp_ns=1, ear=0.05),
        ]
    )
    first = client.post("/api/assess").json()
    assert first["status"] == "ok"
    assert first["features"]["perclos"] == 0.0

    second = client.post("/api/assess").json()
    assert second["status"] == "ok"
    assert second["features"]["perclos"] == 0.5


class FailingPipeline:
    """Raises during ``process_frame`` to exercise error mapping."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def process_frame(self, frame: Frame) -> VisionEvent:
        raise self._error


def test_assess_camera_unavailable_maps_to_503() -> None:
    pipeline = FailingPipeline(CameraNotAvailableError("webcam disconnected"))
    vision = VisionAdapter(
        Settings(),
        camera=FakeCamera(),
        pipeline=pipeline,
    )
    response = TestClient(create_app(vision=vision)).post("/api/assess")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "camera.unavailable"


def test_assess_pipeline_error_maps_to_500() -> None:
    pipeline = FailingPipeline(ProcessingError("model failed to load"))
    vision = VisionAdapter(
        Settings(),
        camera=FakeCamera(),
        pipeline=pipeline,
    )
    response = TestClient(create_app(vision=vision)).post("/api/assess")
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "pipeline.error"