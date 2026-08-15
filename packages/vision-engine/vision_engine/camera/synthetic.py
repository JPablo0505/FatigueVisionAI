"""Deterministic synthetic camera source for tests, CI and development.

Produces reproducible BGR frames without any camera hardware or cv2, so the
rest of the engine can be exercised on every machine.
"""

import time

import numpy as np

from vision_engine.camera.settings import CameraSettings
from vision_engine.errors import CameraNotAvailableError
from vision_engine.types import Frame


class SyntheticCameraSource:
    """``CameraSource`` implementation generating static, noisy frames.

    The base pattern is seeded with a fixed RNG so two instances with the
    same settings produce identical frames, and a frame counter is encoded
    into a corner pixel so streams can be verified visually.
    """

    def __init__(self, settings: CameraSettings | None = None) -> None:
        """Precompute the deterministic base pattern from the settings."""
        self._settings = settings or CameraSettings()
        rng = np.random.default_rng(0)
        self._base = rng.integers(
            0, 256, size=(self._settings.height, self._settings.width, 3)
        ).astype(np.uint8)
        self._frame_count = 0
        self._closed = False

    def read(self) -> Frame:
        """Return the next frame with the frame counter encoded in a corner."""
        if self._closed:
            raise CameraNotAvailableError(
                "Cannot read from a closed camera source"
            )
        frame = self._base.copy()
        frame[0, 0, 0] = self._frame_count % 256
        frame[0, self._settings.width - 1, 0] = self._frame_count % 256
        self._frame_count += 1
        return Frame(data=frame, timestamp_ns=time.time_ns())

    def close(self) -> None:
        """Mark the source as closed; there is no real resource to release."""
        self._closed = True

    @property
    def closed(self) -> bool:
        """Return whether the source has been closed."""
        return self._closed