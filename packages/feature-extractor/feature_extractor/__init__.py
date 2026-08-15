"""Feature Extractor public API.

The extractor is the seam between the Vision Engine (landmarks) and the AI
Engine (``FatigueFeatures``). The public surface is the pair
``FeatureExtractor`` + ``FeatureExtractorConfig`` (plus ``BiometricEvent``
for the realtime event stream); ``geometry`` and ``window`` remain
importable as submodules (for tests and extension), but consumers should
rely on the public API only.
"""

from feature_extractor.extractor import (
    BiometricEvent,
    FeatureExtractor,
    FeatureExtractorConfig,
)

__all__ = ["BiometricEvent", "FeatureExtractor", "FeatureExtractorConfig"]