"""Ports (interfaces) the AI Engine exposes for external adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ai_engine.types import FatigueFeatures, FatigueResult


@runtime_checkable
class FatigueModel(Protocol):
    """Maps fatigue features to a fatigue score.

    Resource management (loading weights, opening session, releasing memory)
    is the responsibility of each implementation, not of this protocol.
    """

    def predict(self, features: FatigueFeatures) -> FatigueResult:
        """Return the fatigue assessment for the given features."""
        ...