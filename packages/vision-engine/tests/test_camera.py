"""Tests for camera sources and their factory (synthetic, no hardware)."""

import platform

import cv2
import numpy as np
import pytest

from vision_engine import (
    CameraSettings,
    CameraSource,
    OpenCVCameraSource,
    SyntheticCameraSource,
    create_camera_source,
)
from vision_engine.camera.opencv import _candidate_backends
from vision_engine.errors import CameraNotAvailableError


def test_synthetic_source_reads_frame_with_settings_shape() -> None:
    source = create_camera_source(
        "synthetic", CameraSettings(width=160, height=120)
    )
    frame = source.read()
    assert frame.width == 160
    assert frame.height == 120
    assert frame.channels == 3
    assert frame.data.dtype == np.uint8
    assert frame.timestamp_ns > 0
    source.close()


def test_synthetic_source_frame_counter_changes_data() -> None:
    source = SyntheticCameraSource(CameraSettings(width=32, height=24))
    first = source.read()
    second = source.read()
    assert first.data[0, 0, 0] != second.data[0, 0, 0]
    assert first.data[0, 0, 0] == 0
    assert second.data[0, 0, 0] == 1
    source.close()


def test_synthetic_source_close_semantics() -> None:
    source = SyntheticCameraSource()
    assert not source.closed
    source.close()
    assert source.closed
    with pytest.raises(CameraNotAvailableError):
        source.read()


def test_synthetic_source_deterministic_base() -> None:
    settings = CameraSettings(width=64, height=48)
    first = SyntheticCameraSource(settings).read()
    second = SyntheticCameraSource(settings).read()
    np.testing.assert_array_equal(first.data, second.data)


def test_factory_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        create_camera_source("bogus", CameraSettings())


def test_opencv_source_semantics() -> None:
    settings = CameraSettings(device_index=0, width=320, height=240)
    try:
        source = OpenCVCameraSource(settings)
    except CameraNotAvailableError:
        pytest.skip("no camera available on this machine")
    try:
        frame = source.read()
        assert frame.channels == 3
        if (frame.width, frame.height) != (settings.width, settings.height):
            pytest.skip(
                "camera driver ignored the requested resolution "
                f"({frame.width}x{frame.height} != "
                f"{settings.width}x{settings.height})"
            )
    finally:
        source.close()
    assert source.closed


def test_contract_sources_implement_camera_source() -> None:
    synthetic = SyntheticCameraSource()
    assert isinstance(synthetic, CameraSource)
    try:
        opencv = OpenCVCameraSource(CameraSettings(device_index=0))
    except CameraNotAvailableError:
        pytest.skip("no camera available on this machine")
    else:
        assert isinstance(opencv, CameraSource)
        opencv.close()


def test_windows_backend_candidates_prefer_dshow() -> None:
    candidates = _candidate_backends()
    if platform.system() == "Windows":
        assert candidates[0] == cv2.CAP_DSHOW
        assert {cv2.CAP_DSHOW, cv2.CAP_MSMF, None}.issuperset(candidates)
    else:
        assert candidates == (None,)