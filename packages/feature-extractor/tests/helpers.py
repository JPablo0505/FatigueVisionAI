"""Synthetic FaceLandmarks meshes with fully controlled geometry.

A mesh must hold at least 478 points: the 468 FaceMesh landmarks (0-467)
plus the iris ring points 468-477. We fill every slot and place the
landmarks the extractor reads at hand-picked coordinates so each geometric
measurement is predictable.
Coordinates use arbitrary units — only relative geometry matters.

Layout (image convention, +x right, +y down):
    - Left eye spans (0, 0)-(1, 0); right eye spans (10, right_eye_dy)-
      (11, right_eye_dy), so the eye-center distance (face width) is ~10.
    - Eyelid gap ``vertical`` makes the EAR of that eye equal ``vertical``
      (the six-point layout normalizes it by the unit width).
    - Iris ring centroids sit at the eye centers plus the requested offset.
    - The nose tip defaults to the eye-line midpoint (symmetric face).
"""

from __future__ import annotations

from vision_engine.types import FaceLandmarks

from feature_extractor.landmarks import (
    LEFT_EYE,
    LEFT_IRIS,
    MOUTH_CORNERS,
    MOUTH_INNER_LIP,
    NOSE_TIP,
    RIGHT_EYE,
    RIGHT_IRIS,
)

_Point = tuple[float, float]

_LANDMARK_COUNT = 478
"""MediaPipe FaceMesh emits 468 facial points (0-467) plus, when the iris
model is enabled, iris ring points 468-477; we allocate the full mesh."""
_LEFT_EYE_BASE_X = 0.0
_RIGHT_EYE_BASE_X = 10.0
_EYE_WIDTH = 1.0
_EYE_LEFT_CENTER_X = 0.5
_EYE_RIGHT_CENTER_X = 10.5
_MOUTH_X = 20.0


def _eye_points(
    base_x: float, base_y: float, width: float, vertical: float
) -> list[_Point]:
    """Six eye points in the canonical EAR ordering (see landmarks module).

    With this layout ``EAR == vertical / width``: the horizontal span between
    corners (elements 0 and 3) is ``width`` and both lid-height pairs
    (elements 1-5 and 2-4) measure ``vertical``.
    """
    inner_x = base_x + 0.25 * width
    outer_x = base_x + 0.75 * width
    return [
        (base_x, base_y),  # p0 outer corner
        (inner_x, base_y - vertical / 2.0),  # p1 upper (inner side)
        (outer_x, base_y - vertical / 2.0),  # p2 upper (outer side)
        (base_x + width, base_y),  # p3 inner corner
        (outer_x, base_y + vertical / 2.0),  # p4 lower (outer side)
        (inner_x, base_y + vertical / 2.0),  # p5 lower (inner side)
    ]


def build_face(
    *,
    left_ear: float = 0.5,
    right_ear: float = 0.5,
    mouth_vertical: float = 0.05,
    mouth_horizontal: float = 1.0,
    iris_offset_x: float = 0.0,
    iris_offset_y: float = 0.0,
    right_eye_dy: float = 0.0,
    nose_x: float | None = None,
    nose_y: float | None = None,
) -> FaceLandmarks:
    """Return a 468-point mesh whose biometrics are controlled by the caller.

    ``left_ear``/``right_ear`` ARE the resulting per-eye EAR values (the
    eyelid gap equals them). ``mouth_vertical``/``mouth_horizontal`` make
    ``MAR == mouth_vertical / mouth_horizontal``. ``right_eye_dy`` shifts
    the right eye down to produce roll. ``iris_offset_*`` displaces both
    iris rings. ``nose_x``/``nose_y`` override the nose tip; the default is
    the eye-line midpoint, which yields a symmetric face (yaw/pitch ~ 0).
    """
    points: list[_Point] = [(0.0, 0.0) for _ in range(_LANDMARK_COUNT)]

    for index, point in zip(
        LEFT_EYE, _eye_points(_LEFT_EYE_BASE_X, 0.0, _EYE_WIDTH, left_ear)
    ):
        points[index] = point
    for index, point in zip(
        RIGHT_EYE, _eye_points(_RIGHT_EYE_BASE_X, right_eye_dy, _EYE_WIDTH, right_ear)
    ):
        points[index] = point

    mid_y = right_eye_dy / 2.0
    mouth_y = mid_y
    points[MOUTH_INNER_LIP[0]] = (_MOUTH_X, mouth_y - mouth_vertical / 2.0)
    points[MOUTH_INNER_LIP[1]] = (_MOUTH_X, mouth_y + mouth_vertical / 2.0)
    points[MOUTH_CORNERS[0]] = (_MOUTH_X - mouth_horizontal / 2.0, mouth_y)
    points[MOUTH_CORNERS[1]] = (_MOUTH_X + mouth_horizontal / 2.0, mouth_y)

    left_iris_x = _EYE_LEFT_CENTER_X + iris_offset_x
    right_iris_x = _EYE_RIGHT_CENTER_X + iris_offset_x
    left_iris_y = 0.0 + iris_offset_y
    right_iris_y = right_eye_dy + iris_offset_y
    for index in LEFT_IRIS:
        points[index] = (left_iris_x, left_iris_y)
    for index in RIGHT_IRIS:
        points[index] = (right_iris_x, right_iris_y)

    eye_mid_x = (_EYE_LEFT_CENTER_X + _EYE_RIGHT_CENTER_X) / 2.0
    points[NOSE_TIP] = (
        eye_mid_x if nose_x is None else nose_x,
        mid_y if nose_y is None else nose_y,
    )

    triplets = tuple((x, y, 0.0) for x, y in points)
    return FaceLandmarks(points=triplets)


def degenerate_face() -> FaceLandmarks:
    """Return a mesh where every landmark collapses to a single point.

    Exercises every division-by-zero guard (eye/mouth width, face width).
    """
    return FaceLandmarks(points=((0.0, 0.0, 0.0),) * _LANDMARK_COUNT)