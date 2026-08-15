"""Deterministic placeholder predictor for the fatigue model contract."""

from __future__ import annotations

from ai_engine.ports import FatigueModel
from ai_engine.types import FatigueFeatures, FatigueResult, default_fatigue_level


class StubFatiguePredictor(FatigueModel):
    """Placeholder — no real fatigue computation.

    Deterministic contract stand-in; replace with a real FatigueModel
    implementation later.
    """

    def predict(self, features: FatigueFeatures) -> FatigueResult:
        """Return a fixed, valid and deterministic fatigue result."""
        score = 0.0
        return FatigueResult(
            score=score,
            level=default_fatigue_level(score),
            confidence=0.0,
            timestamp=None,
        )