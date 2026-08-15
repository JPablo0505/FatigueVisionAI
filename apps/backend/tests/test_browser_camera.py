"""Unit tests for the browser camera source (no MediaPipe, no real device).

``BrowserFrameSource`` is a queue-backed ``CameraSource``: the WebSocket route
submits decoded frames and the streaming loop reads them through the standard
port. These tests cover submit/read, the drop-on-overflow policy, the timeout
that surfaces as ``CameraNotAvailableError`` and the closed semantics.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from vision_engine.errors import CameraNotAvailableError
from vision_engine.types import Frame

from app.adapters.browser_camera import BrowserFrameSource


def _frame(timestamp_ns: int) -> Frame:
    return Frame(data=np.zeros((8, 8, 3), dtype=np.uint8), timestamp_ns=timestamp_ns)


def test_submit_then_read_returns_the_same_frame_data() -> None:
    source = BrowserFrameSource(timeout=0.05)
    frame = _frame(timestamp_ns=42)
    assert source.submit(frame) is True
    got = source.read()
    assert got.timestamp_ns == 42
    assert np.array_equal(got.data, frame.data)


def test_read_blocks_until_a_late_frame_arrives() -> None:
    source = BrowserFrameSource(timeout=2.0)

    def feeder() -> None:
        time.sleep(0.1)
        source.submit(_frame(timestamp_ns=7))

    thread = threading.Thread(target=feeder)
    thread.start()
    try:
        got = source.read()
        assert got.timestamp_ns == 7
    finally:
        thread.join()


def test_read_times_out_with_camera_unavailable() -> None:
    source = BrowserFrameSource(timeout=0.05)
    with pytest.raises(CameraNotAvailableError, match="No se recibieron frames del navegador"):
        source.read()


def test_overflow_drops_newest_frame_and_keeps_oldest() -> None:
    source = BrowserFrameSource(timeout=0.05, maxsize=2)
    assert source.submit(_frame(timestamp_ns=1)) is True
    assert source.submit(_frame(timestamp_ns=2)) is True
    # Queue full: the NEWEST (third) frame is dropped, not the buffered ones.
    assert source.submit(_frame(timestamp_ns=3)) is False
    assert source.read().timestamp_ns == 1
    assert source.read().timestamp_ns == 2
    assert source.queued == 0


def test_open_source_starts_empty() -> None:
    source = BrowserFrameSource(timeout=0.05)
    assert source.closed is False
    assert source.queued == 0


def test_closed_source_rejects_submits_and_reads() -> None:
    source = BrowserFrameSource(timeout=0.05)
    assert source.submit(_frame(timestamp_ns=1)) is True
    source.close()
    assert source.closed is True
    assert source.submit(_frame(timestamp_ns=2)) is False
    with pytest.raises(CameraNotAvailableError, match="cerrada"):
        source.read()


def test_close_is_idempotent() -> None:
    source = BrowserFrameSource(timeout=0.05)
    source.close()
    source.close()
    assert source.closed is True