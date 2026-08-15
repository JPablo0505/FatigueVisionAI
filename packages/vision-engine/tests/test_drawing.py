"""Deterministic tests for the OpenCV drawing helpers (no models needed)."""

import numpy as np

from vision_engine import FaceDetection, FaceLandmarks, Frame, draw_detections, draw_mesh


def make_frame() -> Frame:
    """Return a synthetic 3-channel frame with a fixed timestamp."""
    return Frame(np.zeros((240, 320, 3), dtype=np.uint8), timestamp_ns=7)


def test_draw_detections_keeps_shape_dtype_and_timestamp() -> None:
    frame = make_frame()
    detections = [
        FaceDetection(bbox=(10, 20, 60, 80), score=0.9),
        FaceDetection(bbox=(120, 40, 40, 40), score=0.4),
    ]
    overlay = draw_detections(frame, detections)
    assert overlay.data.shape == frame.data.shape
    assert overlay.data.dtype == frame.data.dtype
    assert overlay.timestamp_ns == frame.timestamp_ns


def test_draw_detections_does_not_mutate_input() -> None:
    frame = make_frame()
    original = frame.data.copy()
    draw_detections(frame, [FaceDetection(bbox=(0, 0, 320, 240), score=1.0)])
    np.testing.assert_array_equal(frame.data, original)


def test_draw_detections_empty_returns_equal_data() -> None:
    frame = make_frame()
    overlay = draw_detections(frame, [])
    np.testing.assert_array_equal(overlay.data, frame.data)


def test_draw_mesh_keeps_shape_dtype_and_timestamp() -> None:
    frame = make_frame()
    meshes = [
        FaceLandmarks(points=((0.1, 0.2, 0.0), (0.5, 0.5, 0.0), (0.9, 0.8, 0.1))),
    ]
    overlay = draw_mesh(frame, meshes)
    assert overlay.data.shape == frame.data.shape
    assert overlay.data.dtype == frame.data.dtype
    assert overlay.timestamp_ns == frame.timestamp_ns


def test_draw_mesh_marks_normalized_corners() -> None:
    frame = make_frame()
    meshes = [
        FaceLandmarks(points=((0.0, 0.0, 0.0), (1.0, 1.0, 0.0))),
    ]
    overlay = draw_mesh(frame, meshes)
    assert overlay.data[0, 0, 0] != frame.data[0, 0, 0]
    assert overlay.data[239, 319, 0] != frame.data[239, 319, 0]


def test_draw_mesh_does_not_mutate_input() -> None:
    frame = make_frame()
    original = frame.data.copy()
    draw_mesh(frame, [FaceLandmarks(points=((0.25, 0.25, 0.0),))])
    np.testing.assert_array_equal(frame.data, original)