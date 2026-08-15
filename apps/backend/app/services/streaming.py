"""Realtime streaming coordinator: one capture loop, many broadcast sockets.

S1-15. The manager owns the ONLY capture loop of the whole application: it
pulls frames, runs them through the Vision pipeline + the shared
``FeatureExtractor`` singleton + the AI engine, and fans the resulting
contract envelopes out to every subscribed WebSocket. A single loop avoids
the OpenCV/MediaPipe camera race that per-client loops would create, and the
extractor singleton remains the sole temporal window in the process.

Pipeline states: ``stopped`` (idle), ``running`` (loop active) and ``error``
(loop paused after a camera/pipeline failure). While ``running`` (or
``error``) a one-second status ticker broadcasts ``vision.status``; an
errored pipeline is retried on a simple 1 s backoff.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress

from ai_engine.errors import InferenceError
from feature_extractor.geometry import (
    GeometryThresholds,
    ear_for,
    head_pose_for,
    iris_direction_for,
    mar_for,
)
from vision_engine.errors import CameraNotAvailableError, ProcessingError
from vision_engine.types import Frame, VisionEvent

from app.adapters.vision import VisionAdapter
from app.api.schemas import AssessResponse
from app.protocol import (
    ERROR_CAMERA_UNAVAILABLE,
    ERROR_INFERENCE,
    ERROR_PIPELINE,
    IncomingMessage,
    MessageKind,
    build_envelope,
    serialize,
)
from app.services.assessment import AssessmentService

STATUS_INTERVAL_S = 1.0
RETRY_BACKOFF_S = 1.0
MAX_ASSESSMENT_HZ = 10.0
OVERLAY_MAX_HZ = 20.0
MAX_EVENTS_PER_ITERATION = 20
QUEUE_BACKLOG = 256

# Umbrales de binarización de geometría facial (S1-18): el backend decide
# "ojos cerrados"/"bostezo" a partir del EAR/MAR puros del feature-extractor.
_GEOMETRY_THRESHOLDS = GeometryThresholds()


def build_face_overlay(event: VisionEvent) -> dict[str, object]:
    """Build the ``FaceOverlay`` payload for one ``VisionEvent``.

    Faces are paired BY INDEX with meshes (the same convention the Feature
    Extractor uses in ``_select_mesh``), taking the shorter of the two lists
    so a detection without a mesh (or vice versa) never crashes the emitter.
    ``face_count`` mirrors the actual number of overlays emitted (==
    ``len(faces)``), which keeps the wire invariant ``face_count ==
    faces.length`` from the contract even if the two lists drift apart.

    The ``bbox`` is derived from the NORMALIZED mesh points (min/max over
    x and y) and NOT from the pixel-space ``FaceDetection.bbox``, which is
    the single normalized source the contract exports.

    S1-18: each face also carries the biometrics derived from the SAME mesh
    so the frontend can highlight them WITHOUT computing anything: the
    ``eyes_closed``/``yawning`` booleans (binarized against the shared
    ``GeometryThresholds``), the normalized ``iris_direction`` and the
    ``head_pose`` in radians. These are additive fields — older clients
    ignore them, newer clients tolerate their absence.
    """
    faces: list[dict[str, object]] = []
    for detection, mesh in zip(event.detections, event.meshes):
        xs = [float(point[0]) for point in mesh.points]
        ys = [float(point[1]) for point in mesh.points]
        dx, dy = iris_direction_for(mesh)
        yaw, pitch, roll = head_pose_for(mesh)
        faces.append(
            {
                "bbox": [
                    min(xs),
                    min(ys),
                    max(xs) - min(xs),
                    max(ys) - min(ys),
                ],
                "score": float(detection.score),
                "points": [
                    [float(point[0]), float(point[1]), float(point[2])]
                    for point in mesh.points
                ],
                "eyes_closed": ear_for(mesh) < _GEOMETRY_THRESHOLDS.ear_closed,
                "yawning": mar_for(mesh) > _GEOMETRY_THRESHOLDS.mar_yawn,
                "iris_direction": {"x": float(dx), "y": float(dy)},
                "head_pose": {
                    "yaw": float(yaw),
                    "pitch": float(pitch),
                    "roll": float(roll),
                },
            }
        )
    return {
        "frame_id": event.frame_id,
        "timestamp_ns": event.timestamp_ns,
        "face_count": len(faces),
        "faces": faces,
    }


class Subscriber:
    """One connected socket plus its bounded outbox queue.

    The manager writes into the queue; the socket's sender task drains it.
    ``alive`` flips to False when the socket breaks so the manager stops
    shipping messages nobody can read.
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=QUEUE_BACKLOG)
        self.alive = True

    def enqueue(self, payload: str) -> None:
        """Queue one already-serialized envelope; drop when backpressure hits."""
        if not self.alive:
            return
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            # Slow consumer: drop. Status and assessments are rate-limited, so
            # the next tick carries fresher data anyway.
            return

    def mark_dead(self) -> None:
        self.alive = False


class StreamingManager:
    """Sole owner of the capture loop; broadcasts contract envelopes.

    State machine: ``stopped`` -> ``running`` <-> ``error``. Control messages
    are GLOBAL (broadcast architecture): ``control.start`` / ``control.stop``
    drive this single loop for every listener (there is only one camera).
    """

    def __init__(self, vision: VisionAdapter, service: AssessmentService) -> None:
        self._vision = vision
        self._service = service
        self._subscribers: set[Subscriber] = set()
        self._loop_task: asyncio.Task[None] | None = None
        self._ticker_task: asyncio.Task[None] | None = None
        self._state = "stopped"
        self._state_error: str | None = None
        self._retry_after = 0.0
        self._last_assessment_at = 0.0
        self._last_overlay_at = 0.0
        self._error_reported = False

    @property
    def state(self) -> str:
        """Current pipeline state: ``stopped`` / ``running`` / ``error``."""
        return self._state

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Bring the capture loop up; also used to retry from ``error``."""
        if self._loop_task is not None and not self._loop_task.done():
            return
        if self._state != "running":
            retrying_after_error = self._state == "error"
            self._state = "running"
            self._state_error = None
            # A fresh start opens a NEW error episode (next failure re-emits
            # the envelope); a backoff retry keeps the current episode silent.
            if not retrying_after_error:
                self._error_reported = False
            await self._broadcast_status()
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Cancel the loop and report the ``stopped`` state.

        Releasing the camera matters as much as stopping the loop: keeping the
        ``VideoCapture`` open leaves the device light on and blocks other
        consumers until the process exits. The adapter drops its cached
        resources, so the next ``start()`` reacquires the camera cleanly.
        """
        await self._cancel_task(self._loop_task)
        self._loop_task = None
        self._error_reported = False
        if self._state != "stopped":
            self._state = "stopped"
            self._state_error = None
            await self._broadcast_status()
        self._vision.close()

    async def shutdown(self) -> None:
        """Cooperative teardown from the app lifespan; idempotent."""
        await self._cancel_task(self._loop_task)
        self._loop_task = None
        await self._cancel_task(self._ticker_task)
        self._ticker_task = None
        self._state = "stopped"
        self._state_error = None
        self._vision.close()

    @staticmethod
    async def _cancel_task(task: asyncio.Task[None] | None) -> None:
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    # -- subscription -------------------------------------------------------

    async def subscribe(self, subscriber: Subscriber) -> None:
        """Register a socket; its very first message is the current status.

        The status ticker is started lazily so it only runs while there is at
        least one listener. The current status is pushed immediately so a
        freshly connected dashboard paints on connect, not after the first
        tick.
        """
        self._subscribers.add(subscriber)
        if self._ticker_task is None or self._ticker_task.done():
            self._ticker_task = asyncio.create_task(self._ticker_loop())
        self._deliver(
            subscriber,
            build_envelope(MessageKind.VISION_STATUS, self._current_status()),
        )

    async def unsubscribe(self, subscriber: Subscriber) -> None:
        """Deregister a socket and stop the loop when the LAST listener leaves.

        Capturing keeps burning camera cycles for nobody after the last socket
        disconnects, so the pipeline drops back to ``stopped``.
        """
        self._subscribers.discard(subscriber)
        subscriber.mark_dead()
        if self._subscribers:
            return
        await self._cancel_task(self._ticker_task)
        self._ticker_task = None
        if self._state == "running":
            await self.stop()

    # -- client message handling ---------------------------------------------

    async def handle_client_message(
        self, subscriber: Subscriber, message: IncomingMessage
    ) -> None:
        """Route one validated client message (ping / control.*)."""
        if message.kind == MessageKind.PING:
            self._deliver(
                subscriber, build_envelope(MessageKind.PONG, {"replyTo": message.message_id})
            )
        elif message.kind == MessageKind.CONTROL_START:
            await self.start()
        elif message.kind == MessageKind.CONTROL_STOP:
            await self.stop()

    def deliver(self, subscriber: Subscriber, envelope: dict[str, object]) -> None:
        """Send one envelope to a single subscriber (used by the WS route)."""
        self._deliver(subscriber, envelope)

    def submit_frame(self, frame: Frame) -> bool:
        """Feed one externally captured frame into the loop's camera source.

        Browser mode: the WebSocket route submits decoded ``frame.provide``
        frames here; ``capture_event()`` reads them from the queue exactly as
        it would read a device frame, so the loop itself does not change.
        """
        return self._vision.submit_frame(frame)

    # -- delivery -----------------------------------------------------------

    async def _broadcast(self, envelope: dict[str, object]) -> None:
        """Serialize once and enqueue to every live subscriber."""
        if not self._subscribers:
            return
        text = serialize(envelope)
        alive = [subscriber for subscriber in self._subscribers if subscriber.alive]
        self._subscribers = set(alive)
        for subscriber in alive:
            subscriber.enqueue(text)

    def _deliver(self, subscriber: Subscriber, envelope: dict[str, object]) -> None:
        subscriber.enqueue(serialize(envelope))

    async def _broadcast_status(self) -> None:
        await self._broadcast(
            build_envelope(MessageKind.VISION_STATUS, self._current_status())
        )

    def _current_status(self) -> dict[str, object]:
        """VisionStatus payload: the adapter's camera facts + manager state.

        The manager owns the streaming pipeline, so ``pipeline``/``error``
        override whatever the one-shot adapter reports (its ``pipeline`` is
        always ``stopped``); camera details come straight from the adapter.
        """
        status = self._service.vision_status()
        status["pipeline"] = self._state
        # Contract: VisionStatus.error is `string | undefined`. Omit the field
        # entirely when there is no error so the wire never carries `null`
        # (the TS parser rejects `error: null` and would drop the whole status).
        # The adapter's status dict always carries `error: None`, so pop it
        # first and only re-add it when the manager holds a real error.
        status.pop("error", None)
        if self._state_error is not None:
            status["error"] = self._state_error
        return status

    # -- background loops -----------------------------------------------------

    async def _ticker_loop(self) -> None:
        """Per-second status beacon; retries an errored pipeline with backoff."""
        while True:
            await asyncio.sleep(STATUS_INTERVAL_S)
            if self._state in ("running", "error"):
                await self._broadcast_status()
            if self._state == "error" and time.monotonic() >= self._retry_after:
                await self.start()

    async def _run_loop(self) -> None:
        """Capture-process-emit loop; runs in its own asyncio task.

        The synchronous chain (camera.read + process_frame + extractor.update +
        predictor) runs in a worker thread so the WebSocket event loop is never
        blocked. The loop drains automatically when a failure moves the state
        to ``error``.
        """
        while self._state == "running":
            response, event = await self._process_one_frame()
            if response is not None:
                # A healthy frame closes the current error episode, so a LATER
                # failure re-emits the ``error`` envelope.
                self._error_reported = False
            await self._maybe_emit_overlay(event)
            await self._maybe_emit_assessment(response)
            await self._maybe_emit_events()

    async def _process_one_frame(self) -> tuple[AssessResponse | None, VisionEvent | None]:
        try:
            return await asyncio.to_thread(self._process_frame_sync)
        except CameraNotAvailableError as exc:
            await self._fail(ERROR_CAMERA_UNAVAILABLE, f"Cámara no disponible: {exc}")
            return None, None
        except ProcessingError as exc:
            await self._fail(ERROR_PIPELINE, f"Error del pipeline de visión: {exc}")
            return None, None
        except InferenceError as exc:
            await self._fail(ERROR_INFERENCE, f"Error de inferencia de fatiga: {exc}")
            return None, None
        except Exception as exc:  # noqa: BLE001 - keep the state machine consistent
            await self._fail(ERROR_PIPELINE, f"Error procesando el frame: {exc}")
            return None, None

    def _process_frame_sync(self) -> tuple[AssessResponse, VisionEvent]:
        """Synchronous chain executed inside the worker thread.

        The ``VisionEvent`` is carried back alongside the assessment so the
        loop can emit the ``face.overlay`` for the SAME frame that produced
        it; the extractor still consumes the event inside ``assess_event``.
        """
        event = self._vision.capture_event()
        return self._service.assess_event(event), event

    async def _maybe_emit_overlay(self, event: VisionEvent | None) -> None:
        """Emit a ``face.overlay`` per frame, capped at ~20 Hz.

        The overlay ALWAYS trails the assessment in the same frame cycle and
        is emitted even with zero faces: the dashboard needs the ``face_count
        == 0`` signal to render "no face detected" over its local video. Only
        the EMISSION is coalesced (separate bucket from assessments) so a
        fast camera cannot flood the wire.
        """
        if event is None:
            return
        now = time.monotonic()
        if now - self._last_overlay_at < (1.0 / OVERLAY_MAX_HZ):
            return
        self._last_overlay_at = now
        await self._broadcast(
            build_envelope(MessageKind.FACE_OVERLAY, build_face_overlay(event))
        )

    async def _maybe_emit_assessment(self, response: AssessResponse | None) -> None:
        """Emit a ``fatigue.assessment`` when features existed, capped at ~10 Hz.

        Every frame still feeds the extractor (the 30 s window needs all of
        them); only the EMISSION is coalesced so a 30 fps camera cannot flood
        the wire. No face in a frame yields no assessment at all — silence is
        the no-face signal (the contract has no ``no_face`` kind).
        """
        if response is None or response.status != "ok" or response.assessment is None:
            return
        now = time.monotonic()
        if now - self._last_assessment_at < (1.0 / MAX_ASSESSMENT_HZ):
            return
        self._last_assessment_at = now
        assessment = response.assessment
        # The wire seals the timestamp in SECONDS before emission: the stub
        # predictor returns timestamp=None (the AI pipeline usually seals it,
        # but this re-seal guarantees a non-null value on the wire).
        timestamp = (
            assessment.timestamp if assessment.timestamp is not None else time.time()
        )
        features = response.features.model_dump() if response.features is not None else None
        payload: dict[str, object] = {
            "score": assessment.score,
            "level": assessment.level,
            "confidence": assessment.confidence,
            "timestamp": timestamp,
            "state": assessment.state,
        }
        if features is not None:
            # Additive contract field (S1-15): the dashboard shows the raw
            # biometrics that fed the evaluation without recomputing anything.
            payload["features"] = features
        await self._broadcast(
            build_envelope(
                MessageKind.FATIGUE_ASSESSMENT,
                payload,
            )
        )

    async def _maybe_emit_events(self) -> None:
        """Broadcast the queued ``biometric.event`` envelopes for this frame.

        The extractor queues one event per completed closure/yawn run; the
        loop drains them every iteration so the dashboard's recent-events
        panel stays fresh without replaying history. The drain is capped per
        iteration so a burst of events cannot flood the wire in one frame —
        the next iterations carry the remainder.
        """
        events = self._service.drain_events()
        for event in events[:MAX_EVENTS_PER_ITERATION]:
            await self._broadcast(
                build_envelope(
                    MessageKind.BIOMETRIC_EVENT,
                    {
                        "type": event.kind,
                        "timestamp": event.timestamp_seconds,
                        "duration_seconds": event.duration_seconds,
                    },
                )
            )

    # -- failure handling ------------------------------------------------------

    async def _fail(self, code: str, message: str) -> None:
        """Drop to ``error``; schedule a retry after ``RETRY_BACKOFF_S``.

        The ``error`` envelope is emitted only once per failure EPISODE: the
        backoff retries keep the status (which carries the ``error`` field)
        but do not spam identical error messages. A healthy frame (or an
        explicit stop/start) opens a new episode.
        """
        first_failure = not self._error_reported
        self._error_reported = True
        self._state = "error"
        self._state_error = message
        self._retry_after = time.monotonic() + RETRY_BACKOFF_S
        if first_failure:
            await self._broadcast(
                build_envelope(MessageKind.ERROR, {"code": code, "message": message})
            )
        await self._broadcast_status()