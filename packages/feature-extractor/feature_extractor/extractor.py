"""Public API of the feature extractor.

The extractor is the SEAM between the Vision Engine and the AI Engine: it
consumes ``VisionEvent`` objects (landmarks) and produces validated
``FatigueFeatures``. It holds no pixels, no media and no camera knowledge;
the only state is the sliding time window that biometric aggregates need.

Tracking scope: SINGLE-SUBJECT tracking — the largest face in each frame
drives the measurements and the rest are ignored. Multi-face tracking is
future work.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_engine.types import FatigueFeatures
from vision_engine.types import FaceLandmarks, VisionEvent

from feature_extractor.geometry import ear_for, head_pose_for, iris_direction_for, mar_for
from feature_extractor.window import (
    BiometricEvent,
    FeatureWindow,
    FrameMeasurements,
    WindowConfig,
)


@dataclass(frozen=True, slots=True)
class FeatureExtractorConfig:
    """Configuration of a ``FeatureExtractor`` instance."""

    window_seconds: float = 5.0
    ear_closed_threshold: float = 0.25
    mar_yawn_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.window_seconds <= 86_400.0:
            raise ValueError(
                f"window_seconds must be in (0, 86400], got {self.window_seconds}"
            )


class FeatureExtractor:
    """Turns ``VisionEvent`` landmarks into sliding-window ``FatigueFeatures``.

    Single-subject tracker (largest face); multi-face is future work. Purely
    synchronous: no I/O, no websockets, no fatigue logic — the AI Engine
    owns the fatigue evaluation. Only per-face geometry and window state
    live here.
    """

    def __init__(self, config: FeatureExtractorConfig | None = None) -> None:
        """Build an extractor; the default config uses a 5-second window."""
        self._config = config if config is not None else FeatureExtractorConfig()
        self._window = FeatureWindow(
            WindowConfig(
                window_seconds=self._config.window_seconds,
                ear_closed_threshold=self._config.ear_closed_threshold,
                mar_yawn_threshold=self._config.mar_yawn_threshold,
            )
        )

    @property
    def window_seconds(self) -> float:
        """Configured window length in seconds (read-only)."""
        return self._config.window_seconds

    def update(self, event: VisionEvent) -> FatigueFeatures | None:
        """Process one event and return the current aggregated features.

        The largest face (largest detection bounding box) sets the frame's
        biometrics. Frames without a usable face are recorded without
        measurements (advancing the window) and yield None. The returned
        features describe the sliding window INCLUDING the current frame;
        the window keeps accumulating afterwards, so ``update`` is safe to
        call repeatedly for a rolling result.
        """
        timestamp_seconds = float(event.timestamp_ns) / 1_000_000_000.0
        mesh = self._select_mesh(event)
        if mesh is None:
            self._window.add_frame(timestamp_seconds, None)
            return None

        iris_x, iris_y = iris_direction_for(mesh)
        yaw, pitch, roll = head_pose_for(mesh)
        # Clamp EAR and MAR to [0, 1] to prevent InvalidFeaturesError from
        # distorted MediaPipe landmarks that push ratios above the logical
        # ceiling (S2-03 debt fix).
        raw_ear = ear_for(mesh)
        raw_mar = mar_for(mesh)
        ear_clamped = max(0.0, min(raw_ear, 1.0))
        mar_clamped = max(0.0, min(raw_mar, 1.0))
        measurements = FrameMeasurements(
            ear=ear_clamped,
            mar=mar_clamped,
            iris_x=iris_x,
            iris_y=iris_y,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )
        self._window.add_frame(timestamp_seconds, measurements)
        return self._window.aggregate()

    def reset(self) -> None:
        """Clear all window state, including events and session counters."""
        self._window.reset()

    def drain_events(self) -> list[BiometricEvent]:
        """Return and clear the queued biometric events, preserving order.

        Events are queued as soon as a closure or yawn run completes in the
        window, so they surface here on the next extraction loop iteration
        (ADITIVO, S1-17). Each event is returned exactly once.
        """
        return self._window.drain_events()

    def _select_mesh(self, event: VisionEvent) -> FaceLandmarks | None:
        """Return the mesh of the largest-detection face, or None.

        Meshes are paired with detections by index, as produced by the
        Vision Pipeline. When there are no detections or meshes, or when the
        largest detection has no matching mesh, the frame is treated as
        having no usable face.
        """
        detections = event.detections
        meshes = event.meshes
        if not detections or not meshes:
            return None
        largest_index = 0
        largest_area = -1.0
        for index, detection in enumerate(detections):
            area = float(detection.bbox[2]) * float(detection.bbox[3])
            if area > largest_area:
                largest_area = area
                largest_index = index
        if largest_index >= len(meshes):
            return None
        return meshes[largest_index]


__all__ = ["BiometricEvent", "FeatureExtractor", "FeatureExtractorConfig"]