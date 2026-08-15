"""AI Engine adapter: thin facade over fatigue assessment."""

from __future__ import annotations

from typing import cast

from ai_engine import FatigueFeatures, FatiguePipeline, StubFatiguePredictor
from ai_engine.ports import FatigueModel


class AIAdapter:
    """Facade over the AI Engine's ``FatiguePipeline``."""

    def __init__(self, predictor: FatigueModel | None = None) -> None:
        self._pipeline = FatiguePipeline(predictor or StubFatiguePredictor())

    def assess(self, features: FatigueFeatures) -> dict[str, object]:
        """Delegate scoring to the engine and return a JSON-clean result."""
        return cast(dict[str, object], self._pipeline.assess(features).to_dict())