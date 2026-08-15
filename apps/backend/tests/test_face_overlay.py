"""Unit tests for ``face.overlay``: payload building and 20 Hz coalescing.

``build_face_overlay`` is the pure payload builder (mesh-derived bbox, score
and full normalized mesh); the emission tests drive ``StreamingManager`` with
the queue-backed pipeline to prove the overlay rides the wire at most ~20 Hz,
in a bucket independent from the ~10 Hz assessments.
"""

from __future__ import annotations

import asyncio
import json
import time
from itertools import pairwise
from typing import cast

from vision_engine.types import FaceDetection, VisionEvent

from app.protocol import MessageKind
from app.services.streaming import Subscriber, build_face_overlay
from tests.streaming_helpers import (
    EventQueuePipeline,
    _drain,
    _pop_until,
    make_manager,
)
from tests.synthetic_faces import _build_mesh, build_face_event, build_no_face_event


def test_build_face_overlay_derives_payload_from_the_mesh() -> None:
    event = build_face_event(frame_id=7, timestamp_ns=123_456_789, ear=0.5, mar=0.05)
    overlay = build_face_overlay(event)

    assert overlay["frame_id"] == 7
    assert overlay["timestamp_ns"] == 123_456_789
    assert overlay["face_count"] == 1

    faces = cast(list[dict[str, object]], overlay["faces"])
    assert len(faces) == 1
    face = faces[0]

    # Score comes from the detection, not the mesh.
    assert face["score"] == event.detections[0].score == 0.9

    # Points are the mesh triplets verbatim (478 landmarks), JSON-clean.
    points = cast(list[list[float]], face["points"])
    assert len(points) == 478
    assert points == [list(point) for point in event.meshes[0].points]

    # bbox is DERIVED from the normalized mesh points (min/max), NOT the
    # pixel-space FaceDetection.bbox.
    mesh_point_xs = [point[0] for point in event.meshes[0].points]
    mesh_point_ys = [point[1] for point in event.meshes[0].points]
    bbox = cast(list[float], face["bbox"])
    assert bbox == [
        min(mesh_point_xs),
        min(mesh_point_ys),
        max(mesh_point_xs) - min(mesh_point_xs),
        max(mesh_point_ys) - min(mesh_point_ys),
    ]
    assert bbox != list(event.detections[0].bbox)

    # S1-18: geometría derivada del mismo mesh. Con ear=0.5 / mar=0.05 el
    # mesh sintético está por encima del umbral de ojos cerrados (0.25) y por
    # debajo del de bostezo (0.5): ambas banderas quedan en False.
    assert face["eyes_closed"] is False
    assert face["yawning"] is False
    iris = cast(dict[str, float], face["iris_direction"])
    assert set(iris.keys()) == {"x", "y"}
    assert all(isinstance(value, float) for value in iris.values())
    head_pose = cast(dict[str, float], face["head_pose"])
    assert set(head_pose.keys()) == {"yaw", "pitch", "roll"}
    assert all(isinstance(value, float) for value in head_pose.values())


def test_build_face_overlay_flags_closed_eyes_and_yawn() -> None:
    # ear=0.1 cierra los ojos (< 0.25); mar=0.8 abre la boca (> 0.5).
    event = build_face_event(frame_id=9, timestamp_ns=1, ear=0.1, mar=0.8)
    overlay = build_face_overlay(event)

    faces = cast(list[dict[str, object]], overlay["faces"])
    assert len(faces) == 1
    assert faces[0]["eyes_closed"] is True
    assert faces[0]["yawning"] is True


def test_build_face_overlay_without_face_is_empty() -> None:
    event = build_no_face_event(frame_id=3, timestamp_ns=0)
    overlay = build_face_overlay(event)

    assert overlay["frame_id"] == 3
    assert overlay["timestamp_ns"] == 0
    assert overlay["face_count"] == 0
    assert overlay["faces"] == []


def test_build_face_overlay_pairs_by_index_taking_the_shorter_list() -> None:
    mesh = _build_mesh(ear=0.4, mar=0.05)
    high = FaceDetection(bbox=(100, 100, 200, 200), score=0.9)
    low = FaceDetection(bbox=(300, 100, 400, 200), score=0.5)

    # More detections than meshes: only the first (index-aligned) face exists.
    more_detections = VisionEvent(
        frame_id=1,
        timestamp_ns=0,
        source="synthetic",
        detections=[high, low],
        meshes=[mesh],
    )
    overlay = build_face_overlay(more_detections)
    assert overlay["face_count"] == 1
    faces = cast(list[dict[str, object]], overlay["faces"])
    assert faces[0]["score"] == 0.9  # paired with meshes[0], not meshes[1]

    # More meshes than detections: still one face (detections drive the count).
    more_meshes = VisionEvent(
        frame_id=2,
        timestamp_ns=0,
        source="synthetic",
        detections=[low],
        meshes=[mesh, mesh],
    )
    overlay = build_face_overlay(more_meshes)
    assert overlay["face_count"] == 1
    faces = cast(list[dict[str, object]], overlay["faces"])
    assert faces[0]["score"] == 0.5


class _TimedSubscriber(Subscriber):
    """Subscriber that records each envelope with its arrival monotonic time."""

    def __init__(self) -> None:
        super().__init__()
        self.received: list[tuple[dict[str, object], float]] = []

    def enqueue(self, payload: str) -> None:
        self.received.append(
            (cast(dict[str, object], json.loads(payload)), time.monotonic())
        )
        super().enqueue(payload)

    @property
    def overlays(self) -> list[tuple[dict[str, object], float]]:
        return [entry for entry in self.received if entry[0]["kind"] == MessageKind.FACE_OVERLAY]


def test_overlay_emission_is_coalesced_to_about_20hz() -> None:
    async def scenario() -> None:
        pipeline = EventQueuePipeline()
        manager = make_manager(pipeline)
        sub = _TimedSubscriber()
        await manager.subscribe(sub)
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)
        await manager.start()
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)

        # Three frames back-to-back (way under 50 ms apart) feed the loop but
        # only the first may emit: the wire is capped at OVERLAY_MAX_HZ = 20.
        for frame_id in (1, 2, 3):
            pipeline.push(build_face_event(frame_id=frame_id, timestamp_ns=frame_id, ear=0.5))
        messages = await _drain(sub, 0.3)
        overlays = [
            message for message in messages if message["kind"] == MessageKind.FACE_OVERLAY
        ]
        assert len(overlays) == 1
        assert overlays[0]["payload"]["frame_id"] == 1
        await manager.shutdown()

    asyncio.run(scenario())


def test_overlay_rate_never_exceeds_twenty_hz() -> None:
    async def scenario() -> None:
        pipeline = EventQueuePipeline()
        manager = make_manager(pipeline)
        sub = _TimedSubscriber()
        await manager.subscribe(sub)
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)
        await manager.start()
        await _pop_until(manager, sub, MessageKind.VISION_STATUS)

        window = 1.5
        for frame_id in range(120):
            pipeline.push(
                build_face_event(frame_id=frame_id, timestamp_ns=frame_id, ear=0.5)
            )
        await _drain(sub, window)

        overlays = sub.overlays
        # A burn of frames must still emit, and never exceed ~20 Hz of traffic.
        assert len(overlays) >= 2
        assert len(overlays) <= int(window * 20) + 3
        # Consecutive emissions are >= 50 ms apart (coalescing gate); the
        # arrival order mirrors the loop's emission order on the same queue.
        timestamps = [arrived for _, arrived in overlays]
        gaps = [later - earlier for earlier, later in pairwise(timestamps)]
        assert gaps and all(gap >= 0.04 for gap in gaps)
        await manager.shutdown()

    asyncio.run(scenario())