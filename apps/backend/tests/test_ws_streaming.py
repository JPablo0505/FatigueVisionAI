"""Route-level WebSocket tests over the ``/ws`` endpoint (no MediaPipe).

Starlette's TestClient drives a real ASGI websocket handshake against the
FastAPI app, so these cover handshake, protocol validation, ping/pong,
control.start/stop and clean disconnection through the full HTTP stack.
"""

from __future__ import annotations

import base64
import time
import uuid
from typing import Any

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.adapters.browser_camera import BrowserFrameSource
from app.adapters.vision import VisionAdapter
from app.config import Settings
from app.di import create_app
from tests.streaming_helpers import EventQueuePipeline, StreamingFakeCamera
from tests.synthetic_faces import build_face_event, build_no_face_event


def _client(pipeline: EventQueuePipeline) -> TestClient:
    vision = VisionAdapter(
        Settings(),
        camera=StreamingFakeCamera(),
        pipeline=pipeline,
    )
    return TestClient(create_app(vision=vision))


def _browser_client(
    pipeline: EventQueuePipeline,
) -> tuple[TestClient, BrowserFrameSource]:
    camera = BrowserFrameSource()
    vision = VisionAdapter(
        Settings(camera_source="browser"),
        camera=camera,
        pipeline=pipeline,
    )
    return TestClient(create_app(vision=vision)), camera


def _jpeg_b64() -> str:
    ok, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _frame_provide_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "frameId": 1,
        "timestamp": time.time(),
        "width": 64,
        "height": 64,
        "jpegBase64": _jpeg_b64(),
    }
    payload.update(overrides)
    return payload


def _envelope(kind: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "messageId": uuid.uuid4().hex,
        "timestamp": time.time(),
        "protocolVersion": "1",
        "kind": kind,
        "payload": payload,
    }


def _read_until_kind(ws: Any, kind: str, max_reads: int = 20) -> Any:
    """Read messages until one of ``kind`` arrives (safety-capped)."""
    for _ in range(max_reads):
        message = ws.receive_json()
        if message["kind"] == kind:
            return message
    raise AssertionError(f"no '{kind}' message arrived within {max_reads} reads")


def test_handshake_pushes_immediate_vision_status() -> None:
    client = _client(EventQueuePipeline())
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert set(first) == {"messageId", "timestamp", "protocolVersion", "kind", "payload"}
        assert first["protocolVersion"] == "1"
        assert first["kind"] == "vision.status"
        # Contract: error is omitted entirely when there is no error.
        assert set(first["payload"]) == {"camera", "pipeline"}
        assert first["payload"]["pipeline"] == "stopped"
        assert first["payload"]["camera"]["source"] == "synthetic"


def test_ping_replies_with_pong_and_correct_reply_to() -> None:
    client = _client(EventQueuePipeline())
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # initial status
        ping = _envelope("ping", {"sentAt": time.time()})
        ws.send_json(ping)
        pong = ws.receive_json()
        assert set(pong) == {"messageId", "timestamp", "protocolVersion", "kind", "payload"}
        assert pong["kind"] == "pong"
        assert pong["payload"] == {"replyTo": ping["messageId"]}


def test_malformed_json_is_protocol_error() -> None:
    client = _client(EventQueuePipeline())
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text("nope{{not-json")
        error = ws.receive_json()
        assert error["kind"] == "error"
        assert set(error["payload"]) == {"code", "message"}
        assert error["payload"]["code"] == "protocol.error"


def test_unknown_kind_is_protocol_error() -> None:
    client = _client(EventQueuePipeline())
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(_envelope("bogus.kind", {}))
        error = ws.receive_json()
        assert error["kind"] == "error"
        assert error["payload"]["code"] == "protocol.error"


def test_bad_protocol_version_is_protocol_error() -> None:
    client = _client(EventQueuePipeline())
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        message = _envelope("ping", {"sentAt": time.time()})
        message["protocolVersion"] = "2"
        ws.send_json(message)
        error = ws.receive_json()
        assert error["kind"] == "error"
        assert error["payload"]["code"] == "protocol.error"


def test_control_start_streams_assessments_and_stop_reports_stopped() -> None:
    pipeline = EventQueuePipeline()
    client = _client(pipeline)
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["payload"]["pipeline"] == "stopped"

        ws.send_json(_envelope("control.start", {}))
        running = ws.receive_json()
        assert running["kind"] == "vision.status"
        assert running["payload"]["pipeline"] == "running"

        pipeline.push(build_face_event(frame_id=1, timestamp_ns=0, ear=0.5))
        assessment = _read_until_kind(ws, "fatigue.assessment")
        assert set(assessment["payload"]) == {
            "score",
            "level",
            "confidence",
            "timestamp",
            "features",
            "state",
        }
        assert isinstance(assessment["payload"]["features"], dict)
        assert isinstance(assessment["payload"]["timestamp"], float)
        assert assessment["payload"]["level"] in {"low", "medium", "high"}
        assert assessment["payload"]["state"] in {
            "normal",
            "long_blink",
            "fatigue_warning",
            "microsleep_suspected",
            "critical_microsleep",
        }

        ws.send_json(_envelope("control.stop", {}))
        stopped = ws.receive_json()
        assert stopped["kind"] == "vision.status"
        assert stopped["payload"]["pipeline"] == "stopped"


def test_control_start_streams_face_overlay() -> None:
    pipeline = EventQueuePipeline()
    client = _client(pipeline)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(_envelope("control.start", {}))
        ws.receive_json()  # running kickoff

        pipeline.push(build_face_event(frame_id=1, timestamp_ns=0, ear=0.5))
        overlay = _read_until_kind(ws, "face.overlay")
        assert set(overlay["payload"]) == {"frame_id", "timestamp_ns", "face_count", "faces"}
        assert overlay["payload"]["frame_id"] == 1
        assert isinstance(overlay["payload"]["timestamp_ns"], int)
        assert overlay["payload"]["face_count"] == 1
        faces = overlay["payload"]["faces"]
        assert len(faces) == 1
        assert set(faces[0]) == {"bbox", "score", "points", "eyes_closed", "yawning", "iris_direction", "head_pose"}
        # 478 MediaPipe landmarks ride the wire; bbox derives from the mesh,
        # NOT from the pixel-space detection (100, 100, 200, 200).
        assert len(faces[0]["points"]) == 478
        assert faces[0]["bbox"] != [100, 100, 200, 200]
        assert faces[0]["score"] == 0.9
        # S1-18: con ear=0.5 / mar=0.05 las banderas quedan apagadas y las
        # métricas llegan como dicts.
        assert faces[0]["eyes_closed"] is False
        assert faces[0]["yawning"] is False
        assert set(faces[0]["iris_direction"]) == {"x", "y"}
        assert set(faces[0]["head_pose"]) == {"yaw", "pitch", "roll"}


def test_frames_without_face_emit_no_assessment_but_status_continues() -> None:
    pipeline = EventQueuePipeline()
    client = _client(pipeline)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(_envelope("control.start", {}))
        ws.receive_json()  # running kickoff

        pipeline.push(build_no_face_event(frame_id=1, timestamp_ns=0))
        seen: list[Any] = []
        statuses = 0
        overlays = 0
        for _ in range(6):
            message = ws.receive_json()
            seen.append(message)
            if message["kind"] == "vision.status":
                statuses += 1
            elif message["kind"] == "face.overlay":
                overlays += 1
            if statuses >= 2 and overlays >= 1:
                break
        # The status ticker keeps flowing and the no-face frame emits a
        # zero-face overlay; assessments never appear (silence = no face).
        assert statuses >= 2
        assert all(message["kind"] != "fatigue.assessment" for message in seen)
        # The zero-face overlay is contract-valid: face_count 0, empty faces,
        # so the dashboard can render "no face detected".
        for message in seen:
            if message["kind"] == "face.overlay":
                assert message["payload"]["face_count"] == 0
                assert message["payload"]["faces"] == []


def test_clean_disconnect_raises_no_exception() -> None:
    pipeline = EventQueuePipeline()
    client = _client(pipeline)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(_envelope("control.start", {}))
        ws.receive_json()
    # Leaving the context closes the socket cleanly; the assertion is surviving.
    assert True


def test_frame_provide_feed_decodes_jpeg_and_streams_overlay() -> None:
    pipeline = EventQueuePipeline()
    client, camera = _browser_client(pipeline)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # initial status
        ws.send_json(_envelope("control.start", {}))
        ws.receive_json()  # running kickoff
        pipeline.push(build_face_event(frame_id=1, timestamp_ns=0, ear=0.5))

        ws.send_json(_envelope("frame.provide", _frame_provide_payload()))
        overlay = _read_until_kind(ws, "face.overlay")
        assert set(overlay["payload"]) == {"frame_id", "timestamp_ns", "face_count", "faces"}
        assert overlay["payload"]["frame_id"] == 1
        assert not camera.closed, "browser source is in use while the loop runs"

    # Leaving the WebSocket (last subscriber) stops the loop; the shared
    # vision.close() releases the browser source exactly like a device.
    assert camera.closed, "disconnect must release the browser camera source"


def test_frame_provide_undecodable_jpeg_emits_camera_unavailable() -> None:
    pipeline = EventQueuePipeline()
    client, _ = _browser_client(pipeline)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # initial status
        # Valid Base64 that is NOT a JPEG: the route treats it as a failed
        # frame (camera.unavailable), never a crash or protocol error.
        payload = _frame_provide_payload(
            jpegBase64=base64.b64encode(b"not-an-image").decode("ascii")
        )
        ws.send_json(_envelope("frame.provide", payload))
        error = ws.receive_json()
        assert error["kind"] == "error"
        assert set(error["payload"]) == {"code", "message"}
        assert error["payload"]["code"] == "camera.unavailable"


def test_frame_provide_invalid_payload_is_protocol_error() -> None:
    pipeline = EventQueuePipeline()
    client, _ = _browser_client(pipeline)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # initial status
        ws.send_json(
            _envelope(
                "frame.provide",
                _frame_provide_payload(width=0),  # zero width violates the shape
            )
        )
        error = ws.receive_json()
        assert error["kind"] == "error"
        assert error["payload"]["code"] == "protocol.error"