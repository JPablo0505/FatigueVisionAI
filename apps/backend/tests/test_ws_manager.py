"""Manager-level unit tests for the streaming coordinator (no MediaPipe).

These drive ``StreamingManager`` directly with queue-backed subscribers: the
state machine, broadcast fan-out, coalescing, retry/backoff and the
"last listener leaves -> stop" rule are proven here deterministically.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest
from vision_engine.errors import CameraNotAvailableError
from vision_engine.types import Frame

from app.protocol import IncomingMessage, MessageKind
from app.services.streaming import Subscriber
from tests.streaming_helpers import (
    AlwaysFailingPipeline,
    EventQueuePipeline,
    FlakyPipeline,
    StreamingFakeCamera,
    _drain,
    _payload,
    _pop_until,
    _read_one,
    make_browser_manager,
    make_manager,
)
from tests.synthetic_faces import build_face_event, build_no_face_event


def test_subscribe_sends_immediate_status_and_broadcast_reaches_all() -> None:
    async def scenario() -> None:
        pipeline = EventQueuePipeline()
        manager = make_manager(pipeline)
        a, b = Subscriber(), Subscriber()
        await manager.subscribe(a)
        await manager.subscribe(b)

        stopped_a, _ = await _pop_until(manager, a, MessageKind.VISION_STATUS)
        stopped_b, _ = await _pop_until(manager, b, MessageKind.VISION_STATUS)
        assert _payload(stopped_a)["pipeline"] == "stopped"
        assert _payload(stopped_b)["pipeline"] == "stopped"
        assert stopped_a["messageId"] != stopped_b["messageId"]

        pipeline.push(build_face_event(frame_id=1, timestamp_ns=0, ear=0.5))
        await manager.start()

        # control.start is GLOBAL: every subscriber sees the running state.
        running_a, _ = await _pop_until(manager, a, MessageKind.VISION_STATUS)
        running_b, _ = await _pop_until(manager, b, MessageKind.VISION_STATUS)
        assert _payload(running_a)["pipeline"] == "running"
        assert _payload(running_b)["pipeline"] == "running"

        # One broadcast -> the SAME envelope reaches every subscriber.
        assessment_a, _ = await _pop_until(manager, a, MessageKind.FATIGUE_ASSESSMENT)
        assessment_b, _ = await _pop_until(manager, b, MessageKind.FATIGUE_ASSESSMENT)
        assert assessment_a["messageId"] == assessment_b["messageId"]
        assert _payload(assessment_a)["timestamp"] == _payload(assessment_b)["timestamp"]

        await manager.stop()
        stopped_a, _ = await _pop_until(manager, a, MessageKind.VISION_STATUS)
        assert _payload(stopped_a)["pipeline"] == "stopped"
        await manager.shutdown()

    asyncio.run(scenario())


def test_assessment_payload_is_sealed_and_contract_shaped() -> None:
    async def scenario() -> None:
        pipeline = EventQueuePipeline()
        manager = make_manager(pipeline)
        sub = Subscriber()
        await manager.subscribe(sub)
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)
        pipeline.push(build_face_event(frame_id=1, timestamp_ns=0, ear=0.5))
        await manager.start()
        assessment, _ = await _pop_until(manager, sub, MessageKind.FATIGUE_ASSESSMENT)
        assert set(_payload(assessment)) == {
            "score",
            "level",
            "confidence",
            "timestamp",
            "features",
            "state",
        }
        assert isinstance(_payload(assessment)["timestamp"], float)
        assert _payload(assessment)["level"] in {"low", "medium", "high"}
        assert _payload(assessment)["state"] in {
            "normal",
            "long_blink",
            "fatigue_warning",
            "microsleep_suspected",
            "critical_microsleep",
        }
        assert isinstance(_payload(assessment)["features"], dict)
        await manager.shutdown()

    asyncio.run(scenario())


def test_assessment_carries_default_risk_state() -> None:
    async def scenario() -> None:
        pipeline = EventQueuePipeline()
        manager = make_manager(pipeline)
        sub = Subscriber()
        await manager.subscribe(sub)
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)
        pipeline.push(build_face_event(frame_id=1, timestamp_ns=0, ear=0.5))
        await manager.start()
        assessment, _ = await _pop_until(manager, sub, MessageKind.FATIGUE_ASSESSMENT)
        # The stub predictor produces clean features, so the risk state is the
        # default "normal" on the wire.
        assert _payload(assessment)["state"] == "normal"
        await manager.shutdown()

    asyncio.run(scenario())


def test_biometric_event_broadcast_on_closure_completion() -> None:
    async def scenario() -> None:
        pipeline = EventQueuePipeline()
        manager = make_manager(pipeline)
        sub = Subscriber()
        await manager.subscribe(sub)
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)
        # Five consecutive closed-eye frames complete a ~0.167 s closure when
        # the next open frame arrives: a "blink" biometric event.
        for frame_id in range(5):
            pipeline.push(
                build_face_event(
                    frame_id=frame_id,
                    timestamp_ns=frame_id * 33_333_333,
                    ear=0.0,
                )
            )
        pipeline.push(
            build_face_event(frame_id=5, timestamp_ns=5 * 33_333_333, ear=0.5)
        )
        await manager.start()
        event, _ = await _pop_until(manager, sub, MessageKind.BIOMETRIC_EVENT)
        payload = _payload(event)
        assert set(payload) == {"type", "timestamp", "duration_seconds"}
        assert payload["type"] == "blink"
        assert isinstance(payload["timestamp"], float)
        assert payload["duration_seconds"] == pytest.approx(5.0 / 30.0, abs=1e-6)
        # Events drain exactly once: nothing more is rebroadcast afterwards.
        remainder = await _drain(sub, 0.2)
        assert all(
            message["kind"] != MessageKind.BIOMETRIC_EVENT for message in remainder
        )
        await manager.shutdown()

    asyncio.run(scenario())


def test_ping_responded_with_pong_reply_to() -> None:
    async def scenario() -> None:
        pipeline = EventQueuePipeline()
        manager = make_manager(pipeline)
        sub = Subscriber()
        await manager.subscribe(sub)
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)

        await manager.handle_client_message(
            sub,
            IncomingMessage(
                message_id="ping-123",
                kind=MessageKind.PING,
                payload={"sentAt": 1.0},
            ),
        )
        pong = await _read_one(sub)
        assert pong["kind"] == MessageKind.PONG
        assert _payload(pong) == {"replyTo": "ping-123"}
        await manager.shutdown()

    asyncio.run(scenario())


def test_last_subscriber_leaving_stops_the_loop() -> None:
    async def scenario() -> None:
        pipeline = EventQueuePipeline()
        manager = make_manager(pipeline)
        sub = Subscriber()
        await manager.subscribe(sub)
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)
        pipeline.push(build_face_event(frame_id=1, timestamp_ns=0, ear=0.5))
        await manager.start()
        await _pop_until(manager, sub, MessageKind.FATIGUE_ASSESSMENT)
        assert manager.state == "running"

        await manager.unsubscribe(sub)
        # No listeners left: capturing makes no sense, drop to stopped.
        assert manager.state == "stopped"
        await manager.shutdown()

    asyncio.run(scenario())


def test_control_stop_prevents_further_assessments() -> None:
    async def scenario() -> None:
        pipeline = EventQueuePipeline()
        manager = make_manager(pipeline)
        sub = Subscriber()
        await manager.subscribe(sub)
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)
        pipeline.push(build_face_event(frame_id=1, timestamp_ns=0, ear=0.5))
        await manager.start()
        await _pop_until(manager, sub, MessageKind.FATIGUE_ASSESSMENT)

        await manager.stop()
        stopped, _ = await _pop_until(manager, sub, MessageKind.VISION_STATUS)
        assert _payload(stopped)["pipeline"] == "stopped"

        # Even a late event cannot produce an assessment once the loop is down.
        pipeline.push(build_face_event(frame_id=2, timestamp_ns=1, ear=0.5))
        rest = await _drain(sub, 0.2)
        assert all(message["kind"] != MessageKind.FATIGUE_ASSESSMENT for message in rest)
        await manager.shutdown()

    asyncio.run(scenario())


def test_stop_releases_camera() -> None:
    async def scenario() -> None:
        camera = StreamingFakeCamera()
        pipeline = EventQueuePipeline()
        manager = make_manager(pipeline, camera=camera)
        sub = Subscriber()
        await manager.subscribe(sub)
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)
        pipeline.push(build_face_event(frame_id=1, timestamp_ns=0, ear=0.5))
        await manager.start()
        await _pop_until(manager, sub, MessageKind.FATIGUE_ASSESSMENT)
        assert not camera.closed, "camera is in use while the pipeline runs"

        # Stopping the loop must release the device (light off): without this,
        # the VideoCapture stays open until the process exits.
        await manager.stop()
        assert camera.closed, "stop must release the camera"
        stop_status, _ = await _pop_until(manager, sub, MessageKind.VISION_STATUS)
        assert _payload(stop_status)["pipeline"] == "stopped"
        await manager.shutdown()

    asyncio.run(scenario())


def test_frames_without_face_emit_no_assessment() -> None:
    async def scenario() -> None:
        pipeline = EventQueuePipeline()
        manager = make_manager(pipeline)
        sub = Subscriber()
        await manager.subscribe(sub)
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)
        await manager.start()
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)  # running kickoff

        pipeline.push(build_no_face_event(frame_id=1, timestamp_ns=0))
        # The periodic status tick keeps flowing; assessments never appear.
        _, seen_first = await _pop_until(manager, sub, MessageKind.VISION_STATUS)
        _, seen_second = await _pop_until(manager, sub, MessageKind.VISION_STATUS)
        assert all(m["kind"] != MessageKind.FATIGUE_ASSESSMENT for m in seen_first)
        assert all(m["kind"] != MessageKind.FATIGUE_ASSESSMENT for m in seen_second)
        await manager.shutdown()

    asyncio.run(scenario())


def test_assessment_emission_is_coalesced_to_about_10hz() -> None:
    async def scenario() -> None:
        pipeline = EventQueuePipeline()
        manager = make_manager(pipeline)
        sub = Subscriber()
        await manager.subscribe(sub)
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)
        await manager.start()
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)

        # Three frames back-to-back (way under 100 ms apart) feed the window
        # but only the first may emit: the wire is capped at MAX_ASSESSMENT_HZ.
        for frame_id in (1, 2, 3):
            pipeline.push(build_face_event(frame_id=frame_id, timestamp_ns=frame_id, ear=0.5))
        messages = await _drain(sub, 0.3)
        assessments = [
            message for message in messages if message["kind"] == MessageKind.FATIGUE_ASSESSMENT
        ]
        assert len(assessments) == 1
        await manager.shutdown()

    asyncio.run(scenario())


def test_error_maps_code_emits_error_once_then_retries_with_backoff() -> None:
    async def scenario() -> None:
        pipeline = FlakyPipeline(failures=1, error=CameraNotAvailableError)
        manager = make_manager(pipeline)
        sub = Subscriber()
        await manager.subscribe(sub)
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)
        await manager.start()

        error_msg, _ = await _pop_until(manager, sub, MessageKind.ERROR)
        assert _payload(error_msg)["code"] == "camera.unavailable"
        assert "message" in _payload(error_msg)
        # The status right after the error reports the error state.
        error_status = await _read_one(sub)
        assert error_status["kind"] == MessageKind.VISION_STATUS
        assert _payload(error_status)["pipeline"] == "error"
        assert manager.state == "error"

        # The 1 s backoff retry succeeds on the next attempt.
        deadline = time.monotonic() + 6.0
        while manager.state != "running" and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        assert manager.state == "running"

        pipeline.push(build_face_event(frame_id=1, timestamp_ns=0, ear=0.5))
        await _pop_until(manager, sub, MessageKind.FATIGUE_ASSESSMENT)
        await manager.shutdown()

    asyncio.run(scenario())


def test_browser_source_feeds_loop_from_submitted_frames() -> None:
    """Browser mode: submit_frame feeds the loop; stop releases the source.

    The manager is built with ``camera_source == "browser"`` but still gets an
    injected BrowserFrameSource camera and a scripted EventQueuePipeline, so
    no MediaPipe is loaded. The frame travels manager.submit_frame ->
    vision.submit_frame -> queue -> loop capture_event(); the fake pipeline
    ignores its content and returns the scripted event deterministically.
    """
    async def scenario() -> None:
        pipeline = EventQueuePipeline()
        manager, camera = make_browser_manager(pipeline)
        sub = Subscriber()
        await manager.subscribe(sub)
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)

        # The scripted pipeline needs its own event ready: the loop drains the
        # browser camera queue first and only then pops this event.
        pipeline.push(build_face_event(frame_id=1, timestamp_ns=0, ear=0.5))
        await manager.start()

        frame = Frame(data=np.zeros((480, 640, 3), dtype=np.uint8), timestamp_ns=42)
        assert manager.submit_frame(frame) is True
        assert not camera.closed, "browser camera is in use while the loop runs"

        overlay, _ = await _pop_until(manager, sub, MessageKind.FACE_OVERLAY)
        assert overlay["kind"] == MessageKind.FACE_OVERLAY
        await _pop_until(manager, sub, MessageKind.FATIGUE_ASSESSMENT)

        # stop() releases the frame source through vision.close(), exactly as
        # it would release a physical device.
        await manager.stop()
        assert camera.closed, "stop must release the browser camera source"
        stop_status, _ = await _pop_until(manager, sub, MessageKind.VISION_STATUS)
        assert _payload(stop_status)["pipeline"] == "stopped"
        await manager.shutdown()

    asyncio.run(scenario())


def test_persistent_failure_emits_only_one_error_message() -> None:
    async def scenario() -> None:
        pipeline = AlwaysFailingPipeline(error=CameraNotAvailableError)
        manager = make_manager(pipeline)
        sub = Subscriber()
        await manager.subscribe(sub)
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)
        await manager.start()

        # First failure emits error + status(error); the 1 s retries keep
        # reporting status but must NOT spam identical error envelopes.
        await _pop_until(manager, sub, MessageKind.ERROR)
        messages = await _drain(sub, 1.6)
        assert all(message["kind"] != MessageKind.ERROR for message in messages)
        assert manager.state == "error"
        await manager.shutdown()

    asyncio.run(scenario())