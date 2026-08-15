"""Synthetic VisionEvents for backend unit tests (no MediaPipe involved).

These meshes reproduce the geometry the real pipeline emits so the feature
extractor can compute EAR/MAR/pose deterministically. The layout mirrors the
helpers shipped in the feature-extractor tests but is kept local so the
backend does not depend on another package's test module.

The mesh holds 478 points: the 468 FaceMesh landmarks (0-467) plus the iris
ring points (468-477). Only relative geometry matters; image convention is
+x right, +y down. With this layout ``EAR == ear`` per eye (unit-width
eyes), and the eye-line midpoint sits on the nose so a symmetric face yields
near-zero yaw/pitch/roll.
"""

from __future__ import annotations

from vision_engine.types import FaceDetection, FaceLandmarks, VisionEvent

_Point = tuple[float, float]

_LANDMARK_COUNT = 478
_LEFT_EYE = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE = (362, 385, 387, 263, 373, 380)
_LEFT_IRIS = (473, 474, 475, 476, 477)
_RIGHT_IRIS = (468, 469, 470, 471, 472)
_MOUTH_INNER_LIP = (13, 14)
_MOUTH_CORNERS = (61, 291)
_NOSE_TIP = 4

_LEFT_EYE_BASE_X = 0.0
_RIGHT_EYE_BASE_X = 10.0
_EYE_WIDTH = 1.0
_LEFT_EYE_CENTER_X = 0.5
_RIGHT_EYE_CENTER_X = 10.5
_MOUTH_X = 20.0


def _eye_points(
    base_x: float, base_y: float, width: float, vertical: float
) -> list[_Point]:
    """Six eye points in the canonical EAR ordering; ``EAR == vertical/width``."""
    inner_x = base_x + 0.25 * width
    outer_x = base_x + 0.75 * width
    return [
        (base_x, base_y),
        (inner_x, base_y - vertical / 2.0),
        (outer_x, base_y - vertical / 2.0),
        (base_x + width, base_y),
        (outer_x, base_y + vertical / 2.0),
        (inner_x, base_y + vertical / 2.0),
    ]


def _build_mesh(*, ear: float, mar: float) -> FaceLandmarks:
    """Return a mesh whose EAR and MAR equal the requested values."""
    points: list[_Point] = [(0.0, 0.0) for _ in range(_LANDMARK_COUNT)]

    for index, point in zip(_LEFT_EYE, _eye_points(_LEFT_EYE_BASE_X, 0.0, _EYE_WIDTH, ear)):
        points[index] = point
    for index, point in zip(_RIGHT_EYE, _eye_points(_RIGHT_EYE_BASE_X, 0.0, _EYE_WIDTH, ear)):
        points[index] = point

    mouth_y = 0.0
    mouth_horizontal = 1.0
    points[_MOUTH_INNER_LIP[0]] = (_MOUTH_X, mouth_y - mar / 2.0)
    points[_MOUTH_INNER_LIP[1]] = (_MOUTH_X, mouth_y + mar / 2.0)
    points[_MOUTH_CORNERS[0]] = (_MOUTH_X - mouth_horizontal / 2.0, mouth_y)
    points[_MOUTH_CORNERS[1]] = (_MOUTH_X + mouth_horizontal / 2.0, mouth_y)

    for index in _LEFT_IRIS:
        points[index] = (_LEFT_EYE_CENTER_X, 0.0)
    for index in _RIGHT_IRIS:
        points[index] = (_RIGHT_EYE_CENTER_X, 0.0)

    eye_mid_x = (_LEFT_EYE_CENTER_X + _RIGHT_EYE_CENTER_X) / 2.0
    points[_NOSE_TIP] = (eye_mid_x, 0.0)

    triplets = tuple((x, y, 0.0) for x, y in points)
    return FaceLandmarks(points=triplets)


def build_face_event(
    *,
    frame_id: int,
    timestamp_ns: int,
    ear: float,
    mar: float = 0.05,
) -> VisionEvent:
    """Return a VisionEvent carrying one face with the requested biometrics.

    ``ear`` below ``0.25`` counts as eyes closed (PERCLOS/blink episodes);
    ``mar`` above ``0.5`` counts as yawning.
    """
    return VisionEvent(
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        source="synthetic",
        detections=[FaceDetection(bbox=(100, 100, 200, 200), score=0.9)],
        meshes=[_build_mesh(ear=ear, mar=mar)],
    )


def build_no_face_event(*, frame_id: int, timestamp_ns: int) -> VisionEvent:
    """Return a VisionEvent with no detections (no usable face)."""
    return VisionEvent(
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        source="synthetic",
        detections=[],
        meshes=[],
    )
