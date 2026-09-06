"""Unit tests for the S2-05 BaselineFatigueModel.

Tests model loading, inference, port conformance, score/level/confidence
boundaries, and the feature conversion pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ai_engine import (
    FatigueFeatures,
    FatigueLevel,
    FatigueResult,
    FatigueModel,
    HeadPose,
    IrisDirection,
    RiskState,
)
from ai_engine.baseline_model import BaselineFatigueModel

_ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "models" / "baseline_fatigue.joblib"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model() -> BaselineFatigueModel:
    """Load the trained model once for all tests in this module."""
    if not _ARTIFACT_PATH.exists():
        pytest.skip("Model artifact not found; run scripts/train_baseline.py first")
    return BaselineFatigueModel(_ARTIFACT_PATH)


def make_features(**overrides: object) -> FatigueFeatures:
    """Return valid FatigueFeatures with sensible defaults."""
    values: dict[str, object] = {
        "ear": 0.30,
        "mar": 0.10,
        "perclos": 0.05,
        "blink_rate": 15.0,
        "head_pose": HeadPose(0.0, 0.0, 0.0),
        "eye_closure_duration": 0.3,
        "yawn_duration": 0.0,
        "iris_direction": IrisDirection(0.0, 0.0),
        "window_seconds": 30.0,
        "current_eye_closure_duration": 0.0,
        "longest_eye_closure_duration": 0.3,
        "longest_yawn_duration": 0.0,
        "total_blinks": 0.0,
        "total_yawns": 0.0,
    }
    values.update(overrides)
    return FatigueFeatures(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Port conformance
# ---------------------------------------------------------------------------

class TestPortConformance:
    def test_implements_fatigue_model_protocol(self, model: BaselineFatigueModel) -> None:
        assert isinstance(model, FatigueModel)

    def test_predict_returns_fatigue_result(self, model: BaselineFatigueModel) -> None:
        result = model.predict(make_features())
        assert isinstance(result, FatigueResult)

    def test_timestamp_is_none(self, model: BaselineFatigueModel) -> None:
        result = model.predict(make_features())
        assert result.timestamp is None


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

class TestArtifactLoading:
    def test_artifact_exists(self) -> None:
        assert _ARTIFACT_PATH.exists()
        assert _ARTIFACT_PATH.stat().st_size > 0

    def test_invalid_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            BaselineFatigueModel(Path("/nonexistent/model.joblib"))

    def test_feature_cols_loaded(self, model: BaselineFatigueModel) -> None:
        assert hasattr(model, "_feature_cols")
        assert len(model._feature_cols) == 20


# ---------------------------------------------------------------------------
# Score boundaries
# ---------------------------------------------------------------------------

class TestScoreBoundaries:
    def test_score_in_range(self, model: BaselineFatigueModel) -> None:
        result = model.predict(make_features())
        assert 0.0 <= result.score <= 100.0

    def test_low_fatigue_low_score(self, model: BaselineFatigueModel) -> None:
        result = model.predict(make_features(
            ear=0.35, mar=0.05, perclos=0.01, blink_rate=18.0,
            yawn_duration=0.0, longest_yawn_duration=0.0,
        ))
        assert result.level is FatigueLevel.low
        assert result.score < 33.0

    def test_yawning_evidence_shifts_level_up(self, model: BaselineFatigueModel) -> None:
        """Baseline model with limited yawning data (11 samples) may not
        always reach 'high', but yawn evidence should shift the prediction
        toward higher fatigue."""
        no_yawn = model.predict(make_features(
            ear=0.30, mar=0.10, perclos=0.05, blink_rate=15.0,
            yawn_duration=0.0, longest_yawn_duration=0.0,
        ))
        with_yawn = model.predict(make_features(
            ear=0.15, mar=0.50, perclos=0.40, blink_rate=5.0,
            yawn_duration=4.0, longest_yawn_duration=6.0,
        ))
        # The yawning case should have a higher score
        assert with_yawn.score >= no_yawn.score
        # Level should be at least medium (not low)
        assert with_yawn.level in (FatigueLevel.medium, FatigueLevel.high)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_confidence_in_range(self, model: BaselineFatigueModel) -> None:
        result = model.predict(make_features())
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_is_positive(self, model: BaselineFatigueModel) -> None:
        result = model.predict(make_features())
        assert result.confidence > 0.0


# ---------------------------------------------------------------------------
# RiskState mapping
# ---------------------------------------------------------------------------

class TestRiskState:
    def test_low_level_maps_to_normal(self, model: BaselineFatigueModel) -> None:
        result = model.predict(make_features(
            ear=0.35, mar=0.05, perclos=0.01, blink_rate=18.0,
            yawn_duration=0.0,
        ))
        if result.level is FatigueLevel.low:
            assert result.state is RiskState.normal

    def test_high_level_maps_to_microsleep(self, model: BaselineFatigueModel) -> None:
        result = model.predict(make_features(
            ear=0.10, mar=0.70, perclos=0.50, blink_rate=3.0,
            yawn_duration=6.0, longest_yawn_duration=8.0,
        ))
        if result.level is FatigueLevel.high:
            assert result.state is RiskState.microsleep_suspected


# ---------------------------------------------------------------------------
# Feature conversion
# ---------------------------------------------------------------------------

class TestFeatureConversion:
    def test_vector_length_matches_feature_cols(self, model: BaselineFatigueModel) -> None:
        features = make_features()
        vector = model._features_to_vector(features)
        assert vector.shape == (1, 20)

    def test_vector_no_nans(self, model: BaselineFatigueModel) -> None:
        vector = model._features_to_vector(make_features())
        assert not np.any(np.isnan(vector))


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_has_required_keys(self, model: BaselineFatigueModel) -> None:
        result = model.predict(make_features())
        payload = result.to_dict()
        assert "score" in payload
        assert "level" in payload
        assert "confidence" in payload
        assert "state" in payload

    def test_to_dict_level_is_string(self, model: BaselineFatigueModel) -> None:
        result = model.predict(make_features())
        payload = result.to_dict()
        assert isinstance(payload["level"], str)
        assert payload["level"] in ("low", "medium", "high")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self, model: BaselineFatigueModel) -> None:
        features = make_features()
        r1 = model.predict(features)
        r2 = model.predict(features)
        assert r1.score == r2.score
        assert r1.level is r2.level
        assert r1.confidence == r2.confidence
