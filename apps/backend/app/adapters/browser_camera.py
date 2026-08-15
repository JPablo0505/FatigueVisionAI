"""Browser camera source: frames arrive over the wire, not from a device.

In ``camera_source == "browser"`` mode the backend never opens a physical
camera: the browser captures with ``getUserMedia``, encodes every frame as a
JPEG and uploads it through ``frame.provide`` WebSocket messages. The route
decodes and submits a prebuilt ``Frame`` here; the streaming loop consumes it
through the ordinary ``CameraSource`` port exactly like a device frame.

The class implements the port structurally (``read``/``close``/``closed``,
like ``OpenCVCameraSource``): no inheritance from the Protocol, matching the
vision engine convention. It contains no MediaPipe or pipeline logic — frames
are built by the WebSocket route.
"""

from __future__ import annotations

import queue

from vision_engine.errors import CameraNotAvailableError
from vision_engine.types import Frame

# How long read() waits for the next browser frame before reporting the camera
# unavailable. Mirrors the pacing of a real device read that stalls.
READ_TIMEOUT_S = 2.0
# Bounded buffer: the browser may burst frames faster than the loop processes
# them, so overflow backs up; the oldest frames are kept and the newest dropped.
MAX_QUEUED_FRAMES = 2

_CLOSED_MESSAGE = "La fuente del navegador está cerrada."
_TIMEOUT_MESSAGE = "No se recibieron frames del navegador."


class BrowserFrameSource:
    """``CameraSource`` fed by ``submit`` instead of by a capture device."""

    def __init__(
        self, *, timeout: float = READ_TIMEOUT_S, maxsize: int = MAX_QUEUED_FRAMES
    ) -> None:
        """Create an empty frame buffer.

        ``timeout`` tunes how long ``read()`` waits for a frame (tests shrink
        it to run fast); ``maxsize`` bounds the queue so a stalled consumer
        drops incoming frames instead of growing memory without limit.
        """
        self._timeout = timeout
        self._queue: queue.Queue[Frame] = queue.Queue(maxsize=maxsize)
        self._closed = False

    def submit(self, frame: Frame) -> bool:
        """Push one frame without blocking; False when dropped.

        The queue is bounded, so a faster producer than consumer backs up:
        on overflow the NEWEST frame is the one dropped (the loop prefers
        older frames so the pipeline stays as current as the buffer allows).
        A closed source also rejects frames.
        """
        if self._closed:
            return False
        try:
            self._queue.put_nowait(frame)
            return True
        except queue.Full:
            return False

    def read(self) -> Frame:
        """Return the next frame, or report the browser source unavailable.

        Blocks up to ``self._timeout`` seconds waiting for a submitted frame;
        a timeout means the browser stopped sending (camera dead) and raises
        ``CameraNotAvailableError`` so the streaming loop enters its usual
        error/retry path. Reading from a closed source also raises.
        """
        if self._closed:
            raise CameraNotAvailableError(_CLOSED_MESSAGE)
        try:
            return self._queue.get(timeout=self._timeout)
        except queue.Empty:
            raise CameraNotAvailableError(_TIMEOUT_MESSAGE) from None

    def close(self) -> None:
        """Mark the source closed; safe to call more than once."""
        self._closed = True

    @property
    def closed(self) -> bool:
        """Return whether this source has been closed."""
        return self._closed

    @property
    def queued(self) -> int:
        """Number of frames waiting to be consumed (diagnostics/tests)."""
        return self._queue.qsize()