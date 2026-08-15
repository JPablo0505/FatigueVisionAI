"""Contract smoke test: every envelope emitted over the wire matches the TS schema.

packages/contracts is the source of truth for the wire. Each JSON emitted by
the backend must have EXACTLY {messageId, timestamp, protocolVersion, kind,
payload}, protocolVersion == "1" and a kind from the TypeScript MessageKind
vocabulary; payload shapes must match the per-kind contracts.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import cast

from vision_engine.errors import CameraNotAvailableError

from app.protocol import IncomingMessage, MessageKind
from app.services.streaming import Subscriber
from tests.streaming_helpers import (
    EventQueuePipeline,
    FlakyPipeline,
    _pop_until,
    make_manager,
)
from tests.synthetic_faces import build_face_event, build_no_face_event

ENVELOPE_KEYS = {"messageId", "timestamp", "protocolVersion", "kind", "payload"}
PROTOCOL_VERSION = "1"
ALLOWED_KINDS = {
    "vision.status",
    "fatigue.assessment",
    "biometric.event",
    "face.overlay",
    "error",
    "ping",
    "pong",
    "control.start",
    "control.stop",
}
VISION_STATUS_KEYS = {"camera", "pipeline"}
VISION_STATUS_KEYS_WITH_ERROR = {"camera", "pipeline", "error"}
FATIGUE_ASSESSMENT_KEYS = {"score", "level", "confidence", "timestamp", "features", "state"}
FATIGUE_ASSESSMENT_STATES = {
    "normal",
    "long_blink",
    "fatigue_warning",
    "microsleep_suspected",
    "critical_microsleep",
}
FATIGUE_FEATURES_KEYS = {
    "ear",
    "mar",
    "perclos",
    "blink_rate",
    "head_pose",
    "eye_closure_duration",
    "yawn_duration",
    "iris_direction",
    "window_seconds",
    "current_yawn_duration",
    "last_yawn_duration",
    "mar_instant",
    "ear_instant",
    "current_eye_closure_duration",
    "longest_eye_closure_duration",
    "longest_yawn_duration",
    "total_blinks",
    "total_yawns",
}
BIOMETRIC_EVENT_KEYS = {"type", "timestamp", "duration_seconds"}
FACE_OVERLAY_KEYS = {"frame_id", "timestamp_ns", "face_count", "faces"}
# S1-18: cada cara del overlay lleva además los biométricos derivados del mesh.
FACE_OVERLAY_FACE_KEYS = {
    "bbox",
    "score",
    "points",
    "eyes_closed",
    "yawning",
    "iris_direction",
    "head_pose",
}
ERROR_PAYLOAD_KEYS = {"code", "message"}
PONG_PAYLOAD_KEYS = {"replyTo"}


def _assert_valid_envelope(message: dict[str, object]) -> None:
    assert set(message) == ENVELOPE_KEYS, f"envelope keys mismatch: {message}"
    assert message["protocolVersion"] == PROTOCOL_VERSION, message
    assert message["kind"] in ALLOWED_KINDS, message
    message_id = message["messageId"]
    timestamp = message["timestamp"]
    assert isinstance(message_id, str) and message_id
    assert isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool)
    assert isinstance(message["payload"], dict)


def _assert_valid_payload(message: dict[str, object]) -> None:
    kind = message["kind"]
    payload = cast(dict[str, object], message["payload"])
    if kind == MessageKind.VISION_STATUS:
        assert payload["pipeline"] in ("running", "stopped", "error")
        camera = cast(dict[str, object], payload["camera"])
        assert set(camera) == {"source", "running", "resolution", "fps"}
        if payload["pipeline"] == "error":
            assert set(payload) == VISION_STATUS_KEYS_WITH_ERROR
            assert payload["error"] is not None
        else:
            # Contract: error is omitted entirely when there is no error.
            assert set(payload) == VISION_STATUS_KEYS
    elif kind == MessageKind.FATIGUE_ASSESSMENT:
        assert set(payload) == FATIGUE_ASSESSMENT_KEYS
        assert payload["level"] in ("low", "medium", "high")
        assert payload["state"] in FATIGUE_ASSESSMENT_STATES
        assert isinstance(payload["timestamp"], float)
        features = cast(dict[str, object], payload["features"])
        assert set(features) == FATIGUE_FEATURES_KEYS
        for key in (
            "ear",
            "mar",
            "perclos",
            "blink_rate",
            "current_yawn_duration",
            "last_yawn_duration",
            "mar_instant",
            "ear_instant",
            "current_eye_closure_duration",
            "longest_eye_closure_duration",
            "longest_yawn_duration",
            "total_blinks",
            "total_yawns",
        ):
            assert isinstance(features[key], float)
        head_pose = cast(dict[str, object], features["head_pose"])
        assert set(head_pose) == {"yaw", "pitch", "roll"}
        iris = cast(dict[str, object], features["iris_direction"])
        assert set(iris) == {"x", "y"}
    elif kind == MessageKind.BIOMETRIC_EVENT:
        assert set(payload) == BIOMETRIC_EVENT_KEYS
        assert payload["type"] in ("blink", "long_blink", "eyes_closed", "yawn")
        assert isinstance(payload["timestamp"], float)
        assert isinstance(payload["duration_seconds"], (int, float))
        assert not isinstance(payload["duration_seconds"], bool)
        assert payload["duration_seconds"] > 0
    elif kind == MessageKind.ERROR:
        assert set(payload) == ERROR_PAYLOAD_KEYS
        assert payload["code"] in {
            "camera.unavailable",
            "pipeline.error",
            "inference.error",
            "protocol.error",
        }
    elif kind == MessageKind.PONG:
        assert set(payload) == PONG_PAYLOAD_KEYS
        assert isinstance(payload["replyTo"], str)
    elif kind == MessageKind.FACE_OVERLAY:
        assert set(payload) == FACE_OVERLAY_KEYS
        assert isinstance(payload["frame_id"], int)
        assert isinstance(payload["timestamp_ns"], int)
        assert not isinstance(payload["timestamp_ns"], bool)
        faces = cast(list[dict[str, object]], payload["faces"])
        assert isinstance(faces, list)
        assert isinstance(payload["face_count"], int)
        assert payload["face_count"] == len(faces)
        for face in faces:
            assert set(face) == FACE_OVERLAY_FACE_KEYS
            bbox = cast(list[object], face["bbox"])
            assert len(bbox) == 4
            assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox)
            assert isinstance(face["score"], (int, float)) and not isinstance(face["score"], bool)
            points = cast(list[list[object]], face["points"])
            assert len(points) > 0
            for point in points:
                assert len(point) == 3
                assert all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) for v in point
                )
            # S1-18: banderas booleanas y métricas dict (sin bools sueltos).
            assert isinstance(face["eyes_closed"], bool)
            assert isinstance(face["yawning"], bool)
            iris = cast(dict[str, object], face["iris_direction"])
            assert set(iris) == {"x", "y"}
            assert all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in iris.values()
            )
            head_pose = cast(dict[str, object], face["head_pose"])
            assert set(head_pose) == {"yaw", "pitch", "roll"}
            assert all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in head_pose.values()
            )


class _RecordingSubscriber(Subscriber):
    """Subscriber that records every envelope the manager delivers."""

    def __init__(self, collected: list[dict[str, object]]) -> None:
        super().__init__()
        self._collected = collected

    def enqueue(self, payload: str) -> None:
        self._collected.append(cast(dict[str, object], json.loads(payload)))
        super().enqueue(payload)


async def _run_scenario() -> list[dict[str, object]]:
    """Drive the full happy+error+no-face path and collect every emitted message."""
    collected: list[dict[str, object]] = []

    # Happy path: start -> status -> assessment -> stop.
    pipeline = EventQueuePipeline()
    manager = make_manager(pipeline)
    sub = _RecordingSubscriber(collected)
    await manager.subscribe(sub)
    await _pop_until(manager, sub, MessageKind.VISION_STATUS)
    pipeline.push(build_face_event(frame_id=1, timestamp_ns=0, ear=0.5))
    await manager.start()
    await _pop_until(manager, sub, MessageKind.FATIGUE_ASSESSMENT)
    await manager.stop()
    await _pop_until(manager, sub, MessageKind.VISION_STATUS)
    await manager.shutdown()

    # No-face path: silence instead of an assessment.
    no_face = EventQueuePipeline()
    manager = make_manager(no_face)
    sub = _RecordingSubscriber(collected)
    await manager.subscribe(sub)
    await _pop_until(manager, sub, MessageKind.VISION_STATUS)
    no_face.push(build_no_face_event(frame_id=1, timestamp_ns=0))
    await manager.start()
    await _pop_until(manager, sub, MessageKind.VISION_STATUS)
    await manager.shutdown()

    # Biometric event path: a completed closure run emits a biometric.event.
    events = EventQueuePipeline()
    manager = make_manager(events)
    sub = _RecordingSubscriber(collected)
    await manager.subscribe(sub)
    await _pop_until(manager, sub, MessageKind.VISION_STATUS)
    for frame_id in range(5):
        events.push(
            build_face_event(
                frame_id=frame_id, timestamp_ns=frame_id * 33_333_333, ear=0.0
            )
        )
    events.push(
        build_face_event(frame_id=5, timestamp_ns=5 * 33_333_333, ear=0.5)
    )
    await manager.start()
    await _pop_until(manager, sub, MessageKind.BIOMETRIC_EVENT)
    await manager.shutdown()

    # Zero-face overlay: a frame with no face still emits face.overlay
    # (FaceOverlay.face_count == 0, faces == []) so the dashboard can show
    # "no face detected"; it must NOT emit an assessment for that frame.
    empty = EventQueuePipeline()
    manager = make_manager(empty)
    sub = _RecordingSubscriber(collected)
    await manager.subscribe(sub)
    await _pop_until(manager, sub, MessageKind.VISION_STATUS)
    empty.push(build_no_face_event(frame_id=2, timestamp_ns=0))
    await manager.start()
    await _pop_until(manager, sub, MessageKind.FACE_OVERLAY)
    await manager.shutdown()

    # Error path: failure emits error + status(error), retry stays silent.
    flaky = FlakyPipeline(failures=1, error=CameraNotAvailableError)
    manager = make_manager(flaky)
    sub = _RecordingSubscriber(collected)
    await manager.subscribe(sub)
    await _pop_until(manager, sub, MessageKind.VISION_STATUS)
    await manager.start()
    await _pop_until(manager, sub, MessageKind.ERROR)
    await manager.shutdown()

    # Pong path: ping -> pong with replyTo + itself a valid envelope.
    pinger = EventQueuePipeline()
    manager = make_manager(pinger)
    sub = _RecordingSubscriber(collected)
    await manager.subscribe(sub)
    await _pop_until(manager, sub, MessageKind.VISION_STATUS)
    await manager.handle_client_message(
        sub,
        IncomingMessage(message_id="abc", kind=MessageKind.PING, payload={"sentAt": time.time()}),
    )
    await _pop_until(manager, sub, MessageKind.PONG)
    await manager.shutdown()

    return collected


def test_every_emitted_message_matches_the_ts_contract() -> None:
    async def scenario() -> None:
        collected = await _run_scenario()
        assert collected, "scenario produced no messages"
        for message in collected:
            _assert_valid_envelope(message)
            _assert_valid_payload(message)

        # The vocabulary was actually exercised: every server kind appeared.
        kinds = {message["kind"] for message in collected}
        assert MessageKind.VISION_STATUS in kinds
        assert MessageKind.FATIGUE_ASSESSMENT in kinds
        assert MessageKind.BIOMETRIC_EVENT in kinds
        assert MessageKind.FACE_OVERLAY in kinds
        assert MessageKind.ERROR in kinds
        assert MessageKind.PONG in kinds
        # Silence = no-face: the no_face episode must NOT have added a kind.
        assert MessageKind.PING not in kinds  # ping is client->server only

    asyncio.run(scenario())