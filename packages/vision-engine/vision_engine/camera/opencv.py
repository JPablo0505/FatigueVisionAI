"""OpenCV camera source backed by ``cv2.VideoCapture``.

Frames are returned in BGR order, the OpenCV convention. A later MediaPipe
stage converts them to RGB; this class contains no business logic.

Backend selection: on Windows, ``cv2.VideoCapture(index)`` alone resolves to
Media Foundation, which cannot read frames while another process (for example
a browser holding ``getUserMedia``) owns the device. DirectShow coexists with
Media Foundation, so Windows candidates prefer DirectShow, then Media
Foundation, then the library default. Each candidate must prove it can read
a real frame before being accepted; a device that opens but never yields a
frame is rejected instead of surfacing later as ``CameraNotAvailableError``.
"""

import platform
import time

import cv2

from vision_engine.camera.settings import CameraSettings
from vision_engine.errors import CameraNotAvailableError
from vision_engine.types import Frame


def _candidate_backends() -> tuple[int | None, ...]:
    """Capture backends to try in order (``None`` = OpenCV default).

    The first candidate that opens and reads a real frame wins; ``None`` is
    kept as the final fallback so non-Windows platforms behave exactly as
    before (default backend only).
    """
    if platform.system() == "Windows":
        return (cv2.CAP_DSHOW, cv2.CAP_MSMF, None)
    return (None,)


class OpenCVCameraSource:
    """``CameraSource`` implementation wrapping a physical or virtual device.

    Implemented structurally against the ``CameraSource`` port: it exposes
    ``read``, ``close`` and ``closed`` without inheriting from the Protocol.
    """

    def __init__(self, settings: CameraSettings) -> None:
        """Open the device and try to apply the requested capture properties."""
        self._capture = self._open_readable(settings)
        self._set_property(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
        self._set_property(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
        self._set_property(cv2.CAP_PROP_FPS, settings.fps)
        self._closed = False

    def _open_readable(self, settings: CameraSettings) -> cv2.VideoCapture:
        """Return the first candidate capture that opens and reads a frame.

        Raises ``CameraNotAvailableError`` when no candidate works, preferring
        the most specific failure message for diagnostics.
        """
        last_error = "No camera backend available"
        for backend in _candidate_backends():
            capture = (
                cv2.VideoCapture(settings.device_index, backend)
                if backend is not None
                else cv2.VideoCapture(settings.device_index)
            )
            if not capture.isOpened():
                capture.release()
                last_error = (
                    f"Could not open camera device at index "
                    f"{settings.device_index}"
                )
                continue
            try:
                success, frame = capture.read()
            except RuntimeError:
                success, frame = False, None
            if not success or frame is None:
                capture.release()
                last_error = (
                    f"Camera opened but could not read a frame (backend "
                    f"{capture.getBackendName()})"
                )
                continue
            return capture
        raise CameraNotAvailableError(last_error)

    def _set_property(self, prop: int, value: int) -> None:
        """Set a capture property, ignoring drivers that reject the value."""
        try:
            self._capture.set(prop, value)
        except RuntimeError:
            pass

    def read(self) -> Frame:
        """Return the next frame, raising if the device is unusable."""
        if self._closed:
            raise CameraNotAvailableError(
                "Cannot read from a closed camera source"
            )
        success, frame = self._capture.read()
        if not success or frame is None:
            raise CameraNotAvailableError(
                "Failed to read a frame (is the camera connected?)"
            )
        return Frame(data=frame, timestamp_ns=time.time_ns())

    def close(self) -> None:
        """Release the underlying capture; safe to call more than once."""
        if self._closed:
            return
        self._capture.release()
        self._closed = True

    @property
    def closed(self) -> bool:
        """Return whether the capture has been released."""
        return self._closed