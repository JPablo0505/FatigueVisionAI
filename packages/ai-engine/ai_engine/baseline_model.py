"""Baseline ML fatigue model adapter.

Loads a scikit-learn artifact (RandomForestClassifier) trained on YAWDD
windowed features and implements the ``FatigueModel`` protocol.

The artifact is a joblib dictionary containing:
- ``model``: fitted RandomForestClassifier
- ``feature_cols``: ordered list of feature column names
- ``target_map``: mapping from fatigue_state to FatigueLevel
- ``class_labels``: list of class labels the model was trained on
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ai_engine.types import (
    FatigueFeatures,
    FatigueLevel,
    FatigueResult,
    HeadPose,
    IrisDirection,
    RiskState,
    default_fatigue_level,
)

# Default artifact path relative to this file's package
_DEFAULT_ARTIFACT = Path(__file__).resolve().parent.parent / "models" / "baseline_fatigue.joblib"


class BaselineFatigueModel:
    """ML baseline fatigue classifier using a trained RandomForest.

    Implements the ``FatigueModel`` protocol by:
    1. Loading the joblib artifact at construction time.
    2. Converting ``FatigueFeatures`` to the feature vector the model expects.
    3. Running ``predict`` + ``predict_proba`` to produce a ``FatigueResult``.
    """

    def __init__(self, artifact_path: str | Path | None = None) -> None:
        """Load the model artifact.

        Parameters
        ----------
        artifact_path : str or Path, optional
            Path to the ``.joblib`` artifact. Defaults to
            ``packages/ai-engine/models/baseline_fatigue.joblib``.
        """
        path = Path(artifact_path) if artifact_path is not None else _DEFAULT_ARTIFACT
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        artifact: dict[str, Any] = joblib.load(path)
        self._model = artifact["model"]
        self._feature_cols: list[str] = artifact["feature_cols"]
        self._class_labels: list[str] = artifact["class_labels"]
        self._target_map: dict[str, str] = artifact.get("target_map", {})

    def predict(self, features: FatigueFeatures) -> FatigueResult:
        """Assess fatigue from windowed biometric features.

        Converts the FatigueFeatures dataclass to the flat feature vector
        the model expects, runs inference, and maps the prediction to a
        FatigueResult with score, level, and confidence.
        """
        x = self._features_to_vector(features)
        prediction = self._model.predict(x)[0]
        probabilities = self._model.predict_proba(x)[0]

        # Map prediction to FatigueLevel
        level = self._prediction_to_level(prediction)

        # Score: map level to 0-100 range
        score = self._level_to_score(level, probabilities)

        # Confidence: max probability
        confidence = float(np.max(probabilities))

        # RiskState: simple mapping from level
        state = self._level_to_risk_state(level)

        return FatigueResult(
            score=score,
            level=level,
            confidence=confidence,
            state=state,
        )

    def _features_to_vector(self, features: FatigueFeatures) -> np.ndarray:
        """Convert FatigueFeatures to a flat numpy array matching training columns."""
        mapping = {
            "ear_mean": features.ear,
            "ear_std": 0.0,  # Not available in FatigueFeatures (window-level only)
            "ear_min": features.ear_instant,
            "ear_max": features.ear,
            "mar_mean": features.mar,
            "mar_std": 0.0,
            "mar_min": features.mar_instant,
            "mar_max": features.mar,
            "perclos": features.perclos,
            "blink_rate": features.blink_rate,
            "eye_closure_mean": features.eye_closure_duration,
            "eye_closure_max": features.longest_eye_closure_duration,
            "yawn_mean": features.yawn_duration,
            "yawn_max": features.longest_yawn_duration,
            "yawn_fraction": min(1.0, features.yawn_duration / max(features.window_seconds, 0.1)),
            "head_pose_yaw_mean": features.head_pose.yaw,
            "head_pose_pitch_mean": features.head_pose.pitch,
            "head_pose_roll_mean": features.head_pose.roll,
            "iris_x_mean": features.iris_direction.x,
            "iris_y_mean": features.iris_direction.y,
        }
        vector = np.array([[mapping.get(col, 0.0) for col in self._feature_cols]])
        return vector

    def _prediction_to_level(self, prediction: str) -> FatigueLevel:
        """Map model prediction string to FatigueLevel."""
        level_map = {
            "low": FatigueLevel.low,
            "medium": FatigueLevel.medium,
            "high": FatigueLevel.high,
        }
        return level_map.get(prediction, FatigueLevel.medium)

    def _level_to_score(self, level: FatigueLevel, probabilities: np.ndarray) -> float:
        """Convert level + probabilities to a 0-100 score."""
        # Base score from level
        base = {
            FatigueLevel.low: 15.0,
            FatigueLevel.medium: 50.0,
            FatigueLevel.high: 85.0,
        }
        return base.get(level, 50.0)

    def _level_to_risk_state(self, level: FatigueLevel) -> RiskState:
        """Map FatigueLevel to RiskState for the realtime dashboard."""
        mapping = {
            FatigueLevel.low: RiskState.normal,
            FatigueLevel.medium: RiskState.fatigue_warning,
            FatigueLevel.high: RiskState.microsleep_suspected,
        }
        return mapping.get(level, RiskState.normal)


__all__ = ["BaselineFatigueModel"]
