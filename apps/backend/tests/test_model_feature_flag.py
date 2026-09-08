"""Tests for S3-01: ML/Heuristic model feature flag and DI wiring.

Validates that:
1. ``Settings.fatigue_model_type`` defaults to ``"ml"`` and accepts env override.
2. ``build_adapters()`` instantiates the correct model class.
3. Both models produce valid ``FatigueResult`` contracts through the pipeline.
4. The full ``create_app()`` wiring works with both model types.
"""

from __future__ import annotations

import pytest
from ai_engine.ports import FatigueModel
from ai_engine.types import FatigueFeatures, HeadPose, IrisDirection

from app.config import Settings
from app.di import _build_fatigue_model, build_adapters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(**overrides: object) -> FatigueFeatures:
    """Build minimal valid FatigueFeatures for testing."""
    defaults = dict(
        ear=0.35, mar=0.05, perclos=0.05, blink_rate=15.0,
        eye_closure_duration=0.1, yawn_duration=0.0,
        current_yawn_duration=0.0, last_yawn_duration=0.0,
        mar_instant=0.05, ear_instant=0.35,
        current_eye_closure_duration=0.0,
        longest_eye_closure_duration=0.2, longest_yawn_duration=0.0,
        total_blinks=5.0, total_yawns=0.0,
        window_seconds=30.0,
        head_pose=HeadPose(yaw=0.0, pitch=0.0, roll=0.0),
        iris_direction=IrisDirection(x=0.0, y=0.0),
    )
    defaults.update(overrides)
    return FatigueFeatures(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Settings / feature flag
# ---------------------------------------------------------------------------

class TestFatigueModelTypeSetting:
    def test_default_is_ml(self) -> None:
        settings = Settings()
        assert settings.fatigue_model_type == "ml"

    def test_env_override_heuristic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FATIGUE_MODEL_TYPE", "heuristic")
        settings = Settings()
        assert settings.fatigue_model_type == "heuristic"

    def test_env_override_ml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FATIGUE_MODEL_TYPE", "ml")
        settings = Settings()
        assert settings.fatigue_model_type == "ml"

    def test_invalid_value_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FATIGUE_MODEL_TYPE", "invalid")
        with pytest.raises(Exception):
            Settings()


# ---------------------------------------------------------------------------
# Model instantiation via DI
# ---------------------------------------------------------------------------

class TestBuildFatigueModel:
    def test_ml_model_instantiated(self) -> None:
        settings = Settings(fatigue_model_type="ml")
        model = _build_fatigue_model(settings)
        assert isinstance(model, FatigueModel)
        # MLFatigueModel has a version property
        assert hasattr(model, "version")
        assert model.version == "fatigue_v3"

    def test_heuristic_model_instantiated(self) -> None:
        settings = Settings(fatigue_model_type="heuristic")
        model = _build_fatigue_model(settings)
        assert isinstance(model, FatigueModel)
        # HeuristicFatigueModel does NOT have a version property
        assert not hasattr(model, "version") or getattr(model, "version", None) != "fatigue_v2"

    def test_invalid_type_raises(self) -> None:
        settings = Settings.__new__(Settings)
        # Manually set invalid value bypassing validation
        object.__setattr__(settings, "fatigue_model_type", "bogus")
        with pytest.raises(ValueError, match="Unknown FATIGUE_MODEL_TYPE"):
            _build_fatigue_model(settings)


# ---------------------------------------------------------------------------
# Both models produce valid FatigueResult through the pipeline
# ---------------------------------------------------------------------------

class TestModelPredictContract:
    def test_ml_predict_returns_valid_result(self) -> None:
        settings = Settings(fatigue_model_type="ml")
        model = _build_fatigue_model(settings)
        result = model.predict(_make_features())
        assert 0.0 <= result.score <= 100.0
        assert result.level.value in ("low", "medium", "high")
        assert 0.0 <= result.confidence <= 1.0
        assert result.timestamp is not None
        assert result.state is not None

    def test_heuristic_predict_returns_valid_result(self) -> None:
        settings = Settings(fatigue_model_type="heuristic")
        model = _build_fatigue_model(settings)
        result = model.predict(_make_features())
        assert 0.0 <= result.score <= 100.0
        assert result.level.value in ("low", "medium", "high")
        assert 0.0 <= result.confidence <= 1.0
        assert result.state is not None


# ---------------------------------------------------------------------------
# build_adapters wires the correct model
# ---------------------------------------------------------------------------

class TestBuildAdapters:
    def test_ml_adapter_uses_ml_model(self) -> None:
        settings = Settings(fatigue_model_type="ml")
        vision, ai = build_adapters(settings)
        # The AIAdapter wraps a FatiguePipeline which holds the predictor
        predictor = ai._pipeline.predictor
        assert hasattr(predictor, "version")
        assert predictor.version == "fatigue_v3"

    def test_heuristic_adapter_uses_heuristic_model(self) -> None:
        settings = Settings(fatigue_model_type="heuristic")
        vision, ai = build_adapters(settings)
        predictor = ai._pipeline.predictor
        # HeuristicFatigueModel has _classify (internal method)
        assert hasattr(predictor, "_classify")

    def test_assess_returns_valid_json(self) -> None:
        """Both model types produce JSON-serializable assessment dicts."""
        for model_type in ("ml", "heuristic"):
            settings = Settings(fatigue_model_type=model_type)
            _, ai = build_adapters(settings)
            result = ai.assess(_make_features())
            assert isinstance(result, dict)
            assert set(result) >= {"score", "level", "confidence", "state"}
            assert 0.0 <= result["score"] <= 100.0
