"""Unit tests for FatiguePipeline orchestration using deterministic fakes."""

import pytest

from ai_engine import (
    FatigueFeatures,
    FatigueLevel,
    FatiguePipeline,
    FatigueResult,
    HeadPose,
    InferenceError,
    IrisDirection,
)
from ai_engine.ports import FatigueModel


class ConstantPredictor(FatigueModel):
    """FatigueModel fake returning a fixed score with an optional timestamp."""

    def __init__(self, score: float, timestamp: float | None = None) -> None:
        self._score = score
        self._timestamp = timestamp

    def predict(self, features: FatigueFeatures) -> FatigueResult:
        level: FatigueLevel
        if self._score < 33.0:
            level = FatigueLevel.low
        elif self._score < 66.0:
            level = FatigueLevel.medium
        else:
            level = FatigueLevel.high
        return FatigueResult(
            score=self._score,
            level=level,
            confidence=0.8,
            timestamp=self._timestamp,
        )


class RaisingPredictor(FatigueModel):
    """FatigueModel fake that always blows up during prediction."""

    def predict(self, features: FatigueFeatures) -> FatigueResult:
        raise RuntimeError("boom")


def make_valid_features() -> FatigueFeatures:
    """Return a valid FatigueFeatures instance for pipeline calls."""
    return FatigueFeatures(
        ear=0.32,
        mar=0.18,
        perclos=0.41,
        blink_rate=18.5,
        head_pose=HeadPose(0.1, 0.2, 0.3),
        eye_closure_duration=0.4,
        yawn_duration=1.2,
        iris_direction=IrisDirection(0.5, -0.25),
        window_seconds=30.0,
    )


def test_assess_seals_missing_timestamp_and_keeps_result() -> None:
    pipeline = FatiguePipeline(ConstantPredictor(score=42.0))
    result = pipeline.assess(make_valid_features())
    assert result.timestamp is not None
    assert result.timestamp > 0
    assert result.score == 42.0
    assert result.level is FatigueLevel.medium
    assert result.confidence == 0.8


def test_assess_keeps_existing_predictor_timestamp() -> None:
    pipeline = FatiguePipeline(ConstantPredictor(score=10.0, timestamp=1234.5))
    result = pipeline.assess(make_valid_features())
    assert result.timestamp == 1234.5


def test_swapping_predictor_changes_behavior() -> None:
    pipeline = FatiguePipeline(ConstantPredictor(score=10.0))
    assert pipeline.assess(make_valid_features()).score == 10.0
    pipeline.predictor = ConstantPredictor(score=90.0)
    assert pipeline.assess(make_valid_features()).score == 90.0


def test_predictor_failure_is_wrapped_in_inference_error() -> None:
    pipeline = FatiguePipeline(RaisingPredictor())
    with pytest.raises(InferenceError):
        pipeline.assess(make_valid_features())