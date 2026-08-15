"""Pure per-frame facial geometry measurements (no state).

Every function turns a single ``FaceLandmarks`` mesh into a plain number or
a tuple of numbers. Functions are side-effect free so they compose in
pipelines and are trivial to test in isolation. Only the standard library
(``math``) is used — none of these measurements justify a numpy dependency.

EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) follow the classic
formulations of Soukupová & Čech (2016) adapted to the MediaPipe FaceMesh
indices. The head-pose heuristic and the iris direction are documented
GEOMETRIC APPROXIMATIONS (see their docstrings); they are meant to be
replaced in a future task by a solvePnP-based estimation that uses real
camera intrinsics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, hypot
from typing import cast

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
_LandmarkTuple = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class GeometryThresholds:
    """Thresholds used to binarize geometric measurements.

    Values follow typical figures reported in the eye-blink / yawn
    literature (PERCLOS studies built on EAR) and are deliberately
    configurable: real deployments may calibrate them per subject or per
    camera setup.
    """

    ear_closed: float = 0.25
    mar_yawn: float = 0.5


def _points(landmarks: FaceLandmarks) -> _LandmarkTuple:
    """Return the mesh point triplets, cast at the untyped-engine boundary."""
    return cast(_LandmarkTuple, landmarks.points)


def _xy(landmarks: FaceLandmarks, index: int) -> _Point:
    """Return the (x, y) coordinates of a landmark (z is ignored)."""
    point = _points(landmarks)[index]
    return (float(point[0]), float(point[1]))


def _distance(a: _Point, b: _Point) -> float:
    """Euclidean distance between two 2-D points."""
    return hypot(a[0] - b[0], a[1] - b[1])


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive ``[low, high]`` interval."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def _eye_ear(
    landmarks: FaceLandmarks, eye: tuple[int, int, int, int, int, int]
) -> float:
    """Eye Aspect Ratio for a single eye: ``(v1 + v2) / (2 * h)``.

    ``eye`` is the canonical six-point ordering where ``h`` is the
    corner-to-corner distance (elements 0 and 3) and ``v1``/``v2`` are the
    two lid-height distances (element pairs 1-5 and 2-4). Degenerate
    geometry (an eye with zero width) yields 0.0.
    """
    p0 = _xy(landmarks, eye[0])
    p1 = _xy(landmarks, eye[1])
    p2 = _xy(landmarks, eye[2])
    p3 = _xy(landmarks, eye[3])
    p4 = _xy(landmarks, eye[4])
    p5 = _xy(landmarks, eye[5])
    horizontal = _distance(p0, p3)
    if horizontal == 0.0:
        return 0.0
    v1 = _distance(p1, p5)
    v2 = _distance(p2, p4)
    return (v1 + v2) / (2.0 * horizontal)


def ear_for(landmarks: FaceLandmarks) -> float:
    """Mean Eye Aspect Ratio across both eyes.

    The both-eye average damps single-eye noise (occlusion, false landmarks)
    and is the recommended aggregate; the two eyes of a healthy subject are
    symmetric, so averaging carries little information cost. Degenerate eyes
    collapse to 0.0.
    """
    left = _eye_ear(landmarks, LEFT_EYE)
    right = _eye_ear(landmarks, RIGHT_EYE)
    return (left + right) / 2.0


def mar_for(landmarks: FaceLandmarks) -> float:
    """Mouth Aspect Ratio: vertical lip opening over horizontal mouth width.

    ``MAR = dist(13, 14) / dist(61, 291)``. A closed mouth yields a small
    positive value; a widely open mouth (yawn) pushes it well above
    ``GeometryThresholds.mar_yawn``. Degenerate geometry (zero mouth width)
    yields 0.0.
    """
    top = _xy(landmarks, MOUTH_INNER_LIP[0])
    bottom = _xy(landmarks, MOUTH_INNER_LIP[1])
    left = _xy(landmarks, MOUTH_CORNERS[0])
    right = _xy(landmarks, MOUTH_CORNERS[1])
    width = _distance(left, right)
    if width == 0.0:
        return 0.0
    return _distance(top, bottom) / width


def _eye_center(
    landmarks: FaceLandmarks, eye: tuple[int, int, int, int, int, int]
) -> _Point:
    """Centroid of the six eye landmarks (robust to individual corner jitter)."""
    xs = [_xy(landmarks, index)[0] for index in eye]
    ys = [_xy(landmarks, index)[1] for index in eye]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _iris_center(landmarks: FaceLandmarks, iris: tuple[int, ...]) -> _Point:
    """Centroid of the iris ring points."""
    xs = [_xy(landmarks, index)[0] for index in iris]
    ys = [_xy(landmarks, index)[1] for index in iris]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def iris_direction_for(landmarks: FaceLandmarks) -> tuple[float, float]:
    """Normalized iris displacement relative to each eye center.

    For each eye the iris centroid (ring indices 468-472 for the right eye,
    473-477 for the left) is compared against the centroid of the six eye
    landmarks and the offset is normalized by the horizontal eye width. The
    two per-eye directions are averaged and clamped to ``[-1, 1]``.

    Approximation: the iris ring landmarks are treated as a rigid centroid.
    MediaPipe only emits iris points when the dedicated iris model is
    enabled, so this is a documented approximation to be revisited together
    with the real iris model. Orientation follows image coordinates: ``+x``
    toward the image right and ``+y`` toward the image bottom. Eyes with
    zero width are skipped; no usable eye yields ``(0.0, 0.0)``.
    """
    left_eye = _eye_center(landmarks, LEFT_EYE)
    right_eye = _eye_center(landmarks, RIGHT_EYE)
    left_iris = _iris_center(landmarks, LEFT_IRIS)
    right_iris = _iris_center(landmarks, RIGHT_IRIS)
    left_width = _distance(
        _xy(landmarks, LEFT_EYE[0]), _xy(landmarks, LEFT_EYE[3])
    )
    right_width = _distance(
        _xy(landmarks, RIGHT_EYE[0]), _xy(landmarks, RIGHT_EYE[3])
    )

    directions: list[_Point] = []
    if left_width > 0.0:
        directions.append(
            (
                (left_iris[0] - left_eye[0]) / left_width,
                (left_iris[1] - left_eye[1]) / left_width,
            )
        )
    if right_width > 0.0:
        directions.append(
            (
                (right_iris[0] - right_eye[0]) / right_width,
                (right_iris[1] - right_eye[1]) / right_width,
            )
        )
    if not directions:
        return (0.0, 0.0)

    dx = sum(direction[0] for direction in directions) / len(directions)
    dy = sum(direction[1] for direction in directions) / len(directions)
    return (_clamp(dx, -1.0, 1.0), _clamp(dy, -1.0, 1.0))


def head_pose_for(landmarks: FaceLandmarks) -> tuple[float, float, float]:
    """Approximate (yaw, pitch, roll) head orientation in RADIANS.

    APPROXIMATE GEOMETRIC HEURISTIC — replace with solvePnP-based estimation
    in a future task. The mesh carries no camera intrinsics, so this maps
    crude image asymmetries onto angles:

    * ``roll``: angle of the eye-center line relative to the image
      horizontal, ``asin((right_eye.y - left_eye.y) / face_width)``
      (positive when the right eye sits lower in the image, i.e. the head
      tilts toward its left shoulder).
    * ``yaw``: horizontal displacement of the nose tip (index 4) from the
      eye-line midpoint, normalized by the face width (positive toward the
      image right).
    * ``pitch``: vertical displacement of the nose tip from the eye-line
      midpoint, normalized by the face width (positive when the nose sits
      lower in the image, i.e. the head pitches down toward the chest).

    Angles are only meaningful within roughly ±0.5 rad; beyond that the
    linear normalization saturates. Degenerate face width (coincident eye
    centers) yields ``(0.0, 0.0, 0.0)``.
    """
    left_eye = _eye_center(landmarks, LEFT_EYE)
    right_eye = _eye_center(landmarks, RIGHT_EYE)
    face_width = _distance(left_eye, right_eye)
    if face_width == 0.0:
        return (0.0, 0.0, 0.0)

    mid_x = (left_eye[0] + right_eye[0]) / 2.0
    mid_y = (left_eye[1] + right_eye[1]) / 2.0
    nose_x, nose_y = _xy(landmarks, NOSE_TIP)

    roll_ratio = _clamp((right_eye[1] - left_eye[1]) / face_width, -1.0, 1.0)
    roll = asin(roll_ratio)
    yaw = (nose_x - mid_x) / face_width
    pitch = (nose_y - mid_y) / face_width
    return (yaw, pitch, roll)


__all__ = ["GeometryThresholds", "ear_for", "head_pose_for", "iris_direction_for", "mar_for"]