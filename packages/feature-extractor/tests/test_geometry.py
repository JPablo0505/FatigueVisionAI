"""Unit tests for the pure geometry functions."""

import math

import pytest

from feature_extractor.geometry import (
    GeometryThresholds,
    ear_for,
    head_pose_for,
    iris_direction_for,
    mar_for,
)
from tests.helpers import build_face, degenerate_face

OPEN_MOUTH_VERTICAL = 0.8
CLOSED_MOUTH_VERTICAL = 0.05


def test_ear_open_eye_is_approximately_half() -> None:
    face = build_face(left_ear=0.5, right_ear=0.5)
    assert ear_for(face) == pytest.approx(0.5, abs=1e-12)


def test_ear_closed_eye_is_zero() -> None:
    face = build_face(left_ear=0.0, right_ear=0.0)
    assert ear_for(face) == pytest.approx(0.0, abs=1e-12)


def test_ear_averages_both_eyes() -> None:
    face = build_face(left_ear=0.5, right_ear=0.0)
    assert ear_for(face) == pytest.approx(0.25, abs=1e-12)


def test_degenerate_mesh_never_raises() -> None:
    face = degenerate_face()
    assert ear_for(face) == pytest.approx(0.0, abs=1e-12)
    assert mar_for(face) == pytest.approx(0.0, abs=1e-12)
    assert head_pose_for(face) == (0.0, 0.0, 0.0)


def test_mar_open_mouth_greater_than_closed() -> None:
    open_mouth = build_face(mouth_vertical=OPEN_MOUTH_VERTICAL)
    closed_mouth = build_face(mouth_vertical=CLOSED_MOUTH_VERTICAL)
    assert mar_for(open_mouth) == pytest.approx(OPEN_MOUTH_VERTICAL, abs=1e-12)
    assert mar_for(closed_mouth) == pytest.approx(CLOSED_MOUTH_VERTICAL, abs=1e-12)
    assert mar_for(open_mouth) > mar_for(closed_mouth)
    assert mar_for(open_mouth) > GeometryThresholds().mar_yawn
    assert mar_for(closed_mouth) < GeometryThresholds().mar_yawn


def test_iris_direction_tracks_offset_and_sign() -> None:
    symmetric = build_face()
    dx_sym, dy_sym = iris_direction_for(symmetric)
    assert (dx_sym, dy_sym) == pytest.approx((0.0, 0.0), abs=1e-12)

    shifted = build_face(iris_offset_x=0.3, iris_offset_y=0.2)
    dx, dy = iris_direction_for(shifted)
    assert dx > 0.0
    assert dy > 0.0
    assert (dx, dy) == pytest.approx((0.3, 0.2), abs=1e-9)
    assert -1.0 <= dx <= 1.0
    assert -1.0 <= dy <= 1.0


def test_iris_direction_clamped_to_unit_range() -> None:
    right = build_face(iris_offset_x=3.0, iris_offset_y=2.0)
    left = build_face(iris_offset_x=-3.0)
    assert iris_direction_for(right) == (1.0, 1.0)
    assert iris_direction_for(left) == (-1.0, 0.0)


def test_head_pose_symmetric_face_is_zero() -> None:
    face = build_face()
    assert head_pose_for(face) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)


def test_head_pose_nose_shift_positive_yaw() -> None:
    face = build_face(nose_x=5.6)
    yaw, pitch, roll = head_pose_for(face)
    assert yaw > 0.0
    assert pitch == pytest.approx(0.0, abs=1e-12)
    assert roll == pytest.approx(0.0, abs=1e-12)


def test_head_pose_eye_tilt_positive_roll() -> None:
    face = build_face(right_eye_dy=0.4)
    yaw, _pitch, roll = head_pose_for(face)
    face_width = math.hypot(10.0, 0.4)
    assert roll == pytest.approx(math.asin(0.4 / face_width), abs=1e-12)
    assert roll > 0.0
    assert yaw == pytest.approx(0.0, abs=1e-12)


def test_geometry_threshold_defaults() -> None:
    thresholds = GeometryThresholds()
    assert thresholds.ear_closed == 0.25
    assert thresholds.mar_yawn == 0.5