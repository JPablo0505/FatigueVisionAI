"""AI Engine adapter: real-time override with head-pose guard.

The ML model classifies on 5-second aggregate windows, which can miss
instantaneous critical events like a 6-second eye closure. This adapter
applies a real-time override AFTER the model prediction.

IMPORTANT GUARD: When the user looks up or down (extreme head pitch),
the camera perspective makes the eyes APPEAR closed even when they're
open. We check head_pose.pitch to skip false-positive overrides during
extreme tilt angles.
"""

from __future__ import annotations

from typing import cast

from ai_engine import FatigueFeatures, FatiguePipeline, StubFatiguePredictor
from ai_engine.ports import FatigueModel
from ai_engine.types import FatigueResult, RiskState

# Real-time override thresholds.
_ACTIVE_CRITICAL_S = 1.50
_ACTIVE_MICROSLEEP_S = 0.80
_ACTIVE_WARNING_S = 0.50

# Head-pose guard: when pitch exceeds this, eye closure measurements are
# unreliable because the camera perspective distorts the eye landmarks.
# The user is looking up/down (keyboard, ceiling) — NOT closing eyes.
_HEAD_PITCH_GUARD_DEG = 25.0

# Severity order for override comparison.
_OVERRIDE_SEVERITY: dict[RiskState, int] = {
    RiskState.normal: 0,
    RiskState.long_blink: 1,
    RiskState.fatigue_warning: 2,
    RiskState.microsleep_suspected: 3,
    RiskState.critical_microsleep: 4,
}


def _realtime_override(features: FatigueFeatures, model_state: RiskState) -> RiskState:
    """Check instantaneous features and escalate if needed.

    Uses ONLY current_eye_closure_duration (real-time, goes to 0 when eyes
    open). Does NOT use longest_eye_closure_duration (cumulative, never
    decreases — would cause stuck alerts).

    HEAD-POSE GUARD: When the user looks up or down (|pitch| > 25°),
    the camera sees the eyes as "partially closed" even when they're open.
    In that case, we skip the override entirely to avoid false positives.
    """
    # Head-pose guard: skip override during extreme tilt
    pitch = abs(features.head_pose.pitch)
    if pitch > _HEAD_PITCH_GUARD_DEG:
        return model_state

    active = features.current_eye_closure_duration

    override = RiskState.normal
    if active >= _ACTIVE_CRITICAL_S:
        override = RiskState.critical_microsleep
    elif active >= _ACTIVE_MICROSLEEP_S:
        override = RiskState.microsleep_suspected
    elif active >= _ACTIVE_WARNING_S:
        override = RiskState.fatigue_warning

    # Only escalate, never downgrade from the model's prediction
    if _OVERRIDE_SEVERITY.get(override, 0) > _OVERRIDE_SEVERITY.get(model_state, 0):
        return override
    return model_state


class AIAdapter:
    """Facade over the AI Engine's ``FatiguePipeline`` with real-time override."""

    def __init__(self, predictor: FatigueModel | None = None) -> None:
        self._pipeline = FatiguePipeline(predictor or StubFatiguePredictor())

    def assess(self, features: FatigueFeatures) -> dict[str, object]:
        """Run ML prediction, then override for active closures."""
        result: FatigueResult = self._pipeline.assess(features)

        overridden_state = _realtime_override(features, result.state)

        if overridden_state != result.state:
            result = FatigueResult(
                score=result.score,
                level=result.level,
                confidence=result.confidence,
                timestamp=result.timestamp,
                state=overridden_state,
            )

        return cast(dict[str, object], result.to_dict())
