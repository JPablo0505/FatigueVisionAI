"""Unit tests for the AI Engine data contract types."""

import json

import pytest

from ai_engine import (
    FatigueFeatures,
    FatigueLevel,
    FatigueResult,
    HeadPose,
    InvalidFeaturesError,
    IrisDirection,
    default_fatigue_level,
)


def make_valid_features() -> FatigueFeatures:
    """Return a valid FatigueFeatures instance for happy-path assertions."""
    return FatigueFeatures(
        ear=0.32,
        mar=0.18,
        perclos=0.41,
        blink_rate=18.5,
        head_pose=HeadPose(yaw=0.1, pitch=0.2, roll=0.3),
        eye_closure_duration=0.4,
        yawn_duration=1.2,
        iris_direction=IrisDirection(x=0.5, y=-0.25),
        window_seconds=30.0,
    )


def test_features_to_dict_is_json_clean() -> None:
    features = make_valid_features()
    payload = features.to_dict()
    json.dumps(payload)
    assert payload["ear"] == 0.32
    assert payload["head_pose"] == {"yaw": 0.1, "pitch": 0.2, "roll": 0.3}
    assert payload["iris_direction"] == {"x": 0.5, "y": -0.25}
    assert payload["window_seconds"] == 30.0


def test_live_indicators_defaults_are_valid() -> None:
    payload = make_valid_features().to_dict()
    # The two yawn durations default to 0; mar_instant defaults to the
    # strictly-positive epsilon floor (same as the extractor's degenerate case).
    assert payload["current_yawn_duration"] == 0.0
    assert payload["last_yawn_duration"] == 0.0
    assert payload["mar_instant"] == 1e-4


def test_live_indicators_serialize_in_to_dict() -> None:
    features = FatigueFeatures(
        ear=0.32,
        mar=0.18,
        perclos=0.41,
        blink_rate=18.5,
        head_pose=HeadPose(0.1, 0.2, 0.3),
        eye_closure_duration=0.4,
        yawn_duration=1.2,
        iris_direction=IrisDirection(0.5, -0.25),
        window_seconds=30.0,
        current_yawn_duration=0.6,
        last_yawn_duration=1.2,
        mar_instant=0.5,
    )
    payload = features.to_dict()
    json.dumps(payload)
    assert payload["current_yawn_duration"] == 0.6
    assert payload["last_yawn_duration"] == 1.2
    assert payload["mar_instant"] == 0.5


def test_current_yawn_duration_must_be_in_range() -> None:
    with pytest.raises(InvalidFeaturesError):
        FatigueFeatures(
            ear=0.32,
            mar=0.18,
            perclos=0.41,
            blink_rate=18.5,
            head_pose=HeadPose(0.1, 0.2, 0.3),
            eye_closure_duration=0.4,
            yawn_duration=1.2,
            iris_direction=IrisDirection(0.5, -0.25),
            window_seconds=30.0,
            current_yawn_duration=-1.0,
        )


def test_last_yawn_duration_must_be_in_range() -> None:
    with pytest.raises(InvalidFeaturesError):
        FatigueFeatures(
            ear=0.32,
            mar=0.18,
            perclos=0.41,
            blink_rate=18.5,
            head_pose=HeadPose(0.1, 0.2, 0.3),
            eye_closure_duration=0.4,
            yawn_duration=1.2,
            iris_direction=IrisDirection(0.5, -0.25),
            window_seconds=30.0,
            last_yawn_duration=21.0,
        )


def test_mar_instant_must_be_positive() -> None:
    with pytest.raises(InvalidFeaturesError):
        FatigueFeatures(
            ear=0.32,
            mar=0.18,
            perclos=0.41,
            blink_rate=18.5,
            head_pose=HeadPose(0.1, 0.2, 0.3),
            eye_closure_duration=0.4,
            yawn_duration=1.2,
            iris_direction=IrisDirection(0.5, -0.25),
            window_seconds=30.0,
            mar_instant=0.0,
        )


def test_ear_must_be_positive() -> None:
    with pytest.raises(InvalidFeaturesError):
        FatigueFeatures(
            ear=0.0,
            mar=0.18,
            perclos=0.41,
            blink_rate=18.5,
            head_pose=HeadPose(0.1, 0.2, 0.3),
            eye_closure_duration=0.4,
            yawn_duration=1.2,
            iris_direction=IrisDirection(0.5, -0.25),
            window_seconds=30.0,
        )


def test_perclos_must_be_in_unit_range() -> None:
    with pytest.raises(InvalidFeaturesError):
        FatigueFeatures(
            ear=0.32,
            mar=0.18,
            perclos=1.5,
            blink_rate=18.5,
            head_pose=HeadPose(0.1, 0.2, 0.3),
            eye_closure_duration=0.4,
            yawn_duration=1.2,
            iris_direction=IrisDirection(0.5, -0.25),
            window_seconds=30.0,
        )


def test_iris_direction_must_be_normalized() -> None:
    with pytest.raises(InvalidFeaturesError):
        FatigueFeatures(
            ear=0.32,
            mar=0.18,
            perclos=0.41,
            blink_rate=18.5,
            head_pose=HeadPose(0.1, 0.2, 0.3),
            eye_closure_duration=0.4,
            yawn_duration=1.2,
            iris_direction=IrisDirection(x=2.0, y=0.0),
            window_seconds=30.0,
        )


def test_window_seconds_must_be_positive() -> None:
    with pytest.raises(InvalidFeaturesError):
        FatigueFeatures(
            ear=0.32,
            mar=0.18,
            perclos=0.41,
            blink_rate=18.5,
            head_pose=HeadPose(0.1, 0.2, 0.3),
            eye_closure_duration=0.4,
            yawn_duration=1.2,
            iris_direction=IrisDirection(0.5, -0.25),
            window_seconds=0.0,
        )


def test_score_must_be_in_range() -> None:
    with pytest.raises(InvalidFeaturesError):
        FatigueResult(score=101.0, level=FatigueLevel.high, confidence=0.9)


def test_confidence_must_be_in_range() -> None:
    with pytest.raises(InvalidFeaturesError):
        FatigueResult(score=50.0, level=FatigueLevel.medium, confidence=1.5)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, FatigueLevel.low),
        (50.0, FatigueLevel.medium),
        (99.0, FatigueLevel.high),
    ],
)
def test_default_fatigue_level_boundaries(score: float, expected: FatigueLevel) -> None:
    assert default_fatigue_level(score) is expected


def test_result_to_dict_with_and_without_timestamp() -> None:
    unsigned = FatigueResult(score=40.0, level=FatigueLevel.medium, confidence=0.7)
    signed = FatigueResult(
        score=40.0, level=FatigueLevel.medium, confidence=0.7, timestamp=1234.5
    )
    json.dumps(unsigned.to_dict())
    json.dumps(signed.to_dict())
    assert unsigned.to_dict()["timestamp"] is None
    assert signed.to_dict()["timestamp"] == 1234.5
    assert signed.to_dict()["level"] == "medium"