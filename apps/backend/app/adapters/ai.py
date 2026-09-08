"""AI Engine adapter: real-time override for instantaneous critical events.

The ML model classifies on 5-second aggregate windows, which can miss
instantaneous critical events like a 6-second eye closure. This adapter
applies a real-time override AFTER the model prediction: if the current
eye closure duration exceeds the critical threshold, the state is forced
to the appropriate risk level regardless of what the model predicted.

IMPORTANT: We only override on `current_eye_closure_duration` (real-time),
NOT on `longest_eye_closure_duration` (cumulative). The longest duration
never decreases within a window, so using it would cause the state to get
STUCK at critical/microsleep even after the user opens their eyes.
"""

from __future__ import annotations

from typing import cast

from ai_engine import FatigueFeatures, FatiguePipeline, StubFatiguePredictor
from ai_engine.ports import FatigueModel
from ai_engine.types import FatigueResult, RiskState

# Real-time override thresholds (same as heuristic model).
_ACTIVE_CRITICAL_S = 1.50
_ACTIVE_MICROSLEEP_S = 0.80
_ACTIVE_WARNING_S = 0.50

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
    """
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
