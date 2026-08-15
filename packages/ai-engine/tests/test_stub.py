"""Unit tests for StubFatiguePredictor determinism and validity."""

from ai_engine import (
    FatigueFeatures,
    FatigueLevel,
    FatigueResult,
    HeadPose,
    IrisDirection,
    StubFatiguePredictor,
)
from ai_engine.ports import FatigueModel


def make_valid_features() -> FatigueFeatures:
    """Return a valid FatigueFeatures instance for stub calls."""
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


def test_stub_implementing_fatigue_model_return_valid_result() -> None:
    predictor = StubFatiguePredictor()
    assert isinstance(predictor, FatigueModel)
    result = predictor.predict(make_valid_features())
    assert isinstance(result, FatigueResult)
    assert 0.0 <= result.score <= 100.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.level is FatigueLevel.low


def test_stub_output_is_deterministic() -> None:
    predictor = StubFatiguePredictor()
    first = predictor.predict(make_valid_features())
    second = predictor.predict(make_valid_features())
    assert first == second


def test_stub_leaves_timestamp_to_pipeline_sealing() -> None:
    result = StubFatiguePredictor().predict(make_valid_features())
    assert result.timestamp is None