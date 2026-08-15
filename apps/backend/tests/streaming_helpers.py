"""Shared fakes and builders for the WebSocket streaming tests.

Everything here avoids MediaPipe: the camera is a stub and the pipeline is
scripted with synthetic VisionEvents, so the manager + route path is
exercised deterministically on any machine.
"""

from __future__ import annotations

import asyncio
import json
import queue as queue_mod
import time
from typing import cast

import numpy as np
from ai_engine import StubFatiguePredictor
from feature_extractor import FeatureExtractor
from vision_engine.types import Frame, VisionEvent

from app.adapters.ai import AIAdapter
from app.adapters.browser_camera import BrowserFrameSource
from app.adapters.vision import VisionAdapter
from app.config import Settings
from app.services.assessment import AssessmentService
from app.services.streaming import StreamingManager, Subscriber

DEFAULT_TIMEOUT = 5.0


def _payload(message: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], message["payload"])


async def _read_one(
    subscriber: Subscriber, timeout: float = DEFAULT_TIMEOUT
) -> dict[str, object]:
    text = await asyncio.wait_for(subscriber.queue.get(), timeout=timeout)
    return cast(dict[str, object], json.loads(text))


async def _pop_until(
    manager: StreamingManager, subscriber: Subscriber, kind: str
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Read messages until one of ``kind`` arrives; return it and everything seen."""
    seen: list[dict[str, object]] = []
    while True:
        message = await _read_one(subscriber)
        seen.append(message)
        if message["kind"] == kind:
            return message, seen


async def _drain(subscriber: Subscriber, window: float) -> list[dict[str, object]]:
    """Read whatever arrives within ``window`` seconds, then give up."""
    messages: list[dict[str, object]] = []
    deadline = time.monotonic() + window
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return messages
        try:
            text = await asyncio.wait_for(subscriber.queue.get(), timeout=remaining)
        except TimeoutError:
            return messages
        messages.append(cast(dict[str, object], json.loads(text)))


class StreamingFakeCamera:
    """Camera that always yields a blank 640x480 frame (no MediaPipe)."""

    def __init__(self) -> None:
        self._closed = False

    def read(self) -> Frame:
        return Frame(data=np.zeros((480, 640, 3), dtype=np.uint8), timestamp_ns=1)

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


class EventQueuePipeline:
    """Scripted pipeline: one synthetic VisionEvent per ``process_frame`` call.

    ``process_frame`` runs in the manager's worker thread, so the feed is a
    thread-safe ``queue.Queue``. It blocks (up to 5 s) while empty so the
    streaming loop waits instead of spinning; tests push events to drive it.
    """

    def __init__(self) -> None:
        self._events: queue_mod.Queue[VisionEvent] = queue_mod.Queue()

    def push(self, event: VisionEvent) -> None:
        self._events.put(event)

    def process_frame(self, frame: Frame) -> VisionEvent:
        return self._events.get(timeout=5.0)


class FlakyPipeline:
    """Fails the first ``failures`` calls, then behaves like ``EventQueuePipeline``.

    Lets tests exercise the error + retry path and then recover.
    """

    def __init__(self, failures: int, error: type[Exception]) -> None:
        self._remaining = failures
        self._error = error
        self._delegate = EventQueuePipeline()

    def push(self, event: VisionEvent) -> None:
        self._delegate.push(event)

    def process_frame(self, frame: Frame) -> VisionEvent:
        if self._remaining > 0:
            self._remaining -= 1
            raise self._error("transient failure")
        return self._delegate.process_frame(frame)


class AlwaysFailingPipeline:
    """Raises the given exception from every ``process_frame`` call."""

    def __init__(self, error: type[Exception]) -> None:
        self._error = error

    def process_frame(self, frame: Frame) -> VisionEvent:
        raise self._error("persistent failure")


def make_manager(
    pipeline: object, *, camera: StreamingFakeCamera | None = None
) -> StreamingManager:
    """Build a StreamingManager with the shared extractor/predictor singletons."""
    vision = VisionAdapter(
        Settings(),
        camera=camera or StreamingFakeCamera(),
        pipeline=pipeline,
    )
    ai = AIAdapter(StubFatiguePredictor())
    service = AssessmentService(vision, ai, FeatureExtractor())
    return StreamingManager(vision, service)


def make_browser_manager(
    pipeline: object, *, camera: BrowserFrameSource | None = None
) -> tuple[StreamingManager, BrowserFrameSource]:
    """Build a StreamingManager in browser-camera mode (no device, no MediaPipe).

    Settings force ``camera_source == "browser"`` — the adapter still gets
    INJECTED fakes (the BrowserFrameSource camera prototype and the scripted
    pipeline) so ``_ensure_resources`` never loads MediaPipe. Returns the
    camera alongside the manager so tests can walk it in/out of ``closed``.
    """
    camera = camera or BrowserFrameSource()
    vision = VisionAdapter(
        Settings(camera_source="browser"),
        camera=camera,
        pipeline=pipeline,
    )
    ai = AIAdapter(StubFatiguePredictor())
    service = AssessmentService(vision, ai, FeatureExtractor())
    return StreamingManager(vision, service), camera