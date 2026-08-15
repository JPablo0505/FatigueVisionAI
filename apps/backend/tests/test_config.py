"""Tests for application settings."""

import pytest

from app.config import REPO_ROOT, Settings


def test_default_model_dir_points_to_repo_models() -> None:
    settings = Settings()
    expected = REPO_ROOT / "models" / "vision"
    assert settings.model_dir == expected
    assert settings.face_detector_model == expected / "face_detector.tflite"
    assert settings.face_landmarker_model == expected / "face_landmarker.task"


def test_env_override_changes_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAMERA_SOURCE", "opencv")
    monkeypatch.setenv("CAMERA_INDEX", "2")
    monkeypatch.setenv("SYNTHETIC_WIDTH", "320")
    settings = Settings()
    assert settings.camera_source == "opencv"
    assert settings.camera_index == 2
    assert settings.synthetic_width == 320