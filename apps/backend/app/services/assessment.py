"""Application service that keeps routes dumb and coordinates adapters.

The backend deliberately computes ZERO biometrics: it reads a
``VisionEvent`` off the Vision adapter, hands it to the shared
``FeatureExtractor``, and forwards the resulting ``FatigueFeatures`` to the
AI adapter. All geometry and temporal-window logic lives in the
``fatiguevision-feature-extractor`` package.
"""

from __future__ import annotations

from typing import cast

from ai_engine.types import FatigueFeatures
from feature_extractor import BiometricEvent, FeatureExtractor
from vision_engine.types import VisionEvent

from app.adapters.ai import AIAdapter
from app.adapters.vision import VisionAdapter
from app.api.schemas import AssessResponse, FatigueFeaturesModel, FatigueResultModel

NO_FACE_MESSAGE = (
    "No usable face in the latest frame; no fatigue features were produced."
)


class AssessmentService:
    """Coordination layer between the Vision and AI adapters.

    The ``FeatureExtractor`` is a SHARED SINGLETON built once in the DI
    container and reused for the whole application lifetime: its 30-second
    sliding window must accumulate frames across endpoint calls, so a fresh
    per-request instance would never aggregate anything. The service never
    constructs extractors itself — it only receives the instance to use.
    """

    def __init__(
        self,
        vision: VisionAdapter,
        ai: AIAdapter,
        extractor: FeatureExtractor,
    ) -> None:
        self._vision = vision
        self._ai = ai
        self._extractor = extractor

    @property
    def extractor(self) -> FeatureExtractor:
        """The shared extractor singleton (its window survives between calls)."""
        return self._extractor

    def drain_events(self) -> list[BiometricEvent]:
        """Drain the completed biometric events queued by the extractor.

        Each completed closure/yawn run in the window queues one event; the
        streaming loop drains them every iteration and broadcasts them as
        ``biometric.event`` envelopes (S1-17).
        """
        return cast(list[BiometricEvent], self._extractor.drain_events())

    def vision_status(self) -> dict[str, object]:
        return self._vision.status()

    def capture_frame(self) -> dict[str, object]:
        return self._vision.capture_once()

    def assess(self) -> AssessResponse:
        """Capture one frame through the configured camera, then assess it."""
        return self.assess_event(self._vision.capture_event())

    def assess_event(self, event: VisionEvent) -> AssessResponse:
        """Run a ``VisionEvent`` through the extractor and the AI engine.

        When the extractor yields features (a usable face was present) they
        are echoed back and sent to the predictor; otherwise the ``no_face``
        envelope is returned and the predictor is never touched.
        """
        features = cast(FatigueFeatures | None, self._extractor.update(event))
        if features is None:
            return AssessResponse(
                status="no_face",
                features=None,
                assessment=None,
                message=NO_FACE_MESSAGE,
            )
        assessment = self._ai.assess(features)
        return AssessResponse(
            status="ok",
            features=FatigueFeaturesModel.model_validate(features.to_dict()),
            assessment=FatigueResultModel.model_validate(assessment),
            message=None,
        )
