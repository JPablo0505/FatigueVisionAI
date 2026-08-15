"""Composition root for deterministic, synchronous fatigue scoring."""

from __future__ import annotations

import time

from ai_engine.errors import InferenceError
from ai_engine.ports import FatigueModel
from ai_engine.types import FatigueFeatures, FatigueResult


class FatiguePipeline:
    """Orchestrates fatigue assessment around an injected ``FatigueModel``.

    The pipeline performs pure orchestration: it delegates scoring to the
    injected predictor and seals the result timestamp when the predictor
    does not provide one. It never computes fatigue itself and never assigns
    the fatigue level. Replacing ``predictor`` with another object that
    implements ``FatigueModel`` swaps the scoring strategy at runtime.
    """

    def __init__(self, predictor: FatigueModel) -> None:
        """Build a pipeline around the given model implementation."""
        self.predictor = predictor

    def assess(self, features: FatigueFeatures) -> FatigueResult:
        """Score the features through the predictor and seal the timestamp.

        The result is returned as-is when the predictor already set the
        timestamp; otherwise it is sealed with the current epoch time.
        """
        try:
            result = self.predictor.predict(features)
        except Exception as exc:
            raise InferenceError(f"Predictor failed: {exc}") from exc
        if result.timestamp is None:
            result = FatigueResult(
                score=result.score,
                level=result.level,
                confidence=result.confidence,
                timestamp=time.time(),
                state=result.state,
            )
        return result