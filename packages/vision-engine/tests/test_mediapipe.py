"""Integration tests for the MediaPipe Tasks adapters.

Model-dependent tests skip when the assets are missing from
``models/vision/`` at the repository root.
"""

from pathlib import Path

import numpy as np
import pytest

from vision_engine import (
    FaceDetection,
    FaceLandmarks,
    Frame,
    MediaPipeFaceDetector,
    MediaPipeFaceMesh,
)
from vision_engine.errors import ProcessingError

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODEL_DIR = _REPO_ROOT / "models" / "vision"
_MODELS = {
    "detector": _MODEL_DIR / "face_detector.tflite",
    "mesh": _MODEL_DIR / "face_landmarker.task",
}


def requires_model(name: str) -> pytest.MarkDecorator:
    """Return a skip marker for tests that need the named model asset."""
    path = _MODELS[name]
    return pytest.mark.skipif(not path.exists(), reason=f"model missing: {path}")


def make_frame() -> Frame:
    """Return a blank BGR frame with a fixed timestamp."""
    return Frame(np.zeros((480, 640, 3), dtype=np.uint8), timestamp_ns=1_000)


def test_detector_missing_model_raises_processing_error() -> None:
    with pytest.raises(ProcessingError):
        MediaPipeFaceDetector("nonexistent.tflite")


def test_mesh_missing_model_raises_processing_error() -> None:
    with pytest.raises(ProcessingError):
        MediaPipeFaceMesh("nonexistent.task")


@requires_model("detector")
def test_detector_returns_face_detection_contract() -> None:
    detector = MediaPipeFaceDetector(str(_MODELS["detector"]))
    try:
        detections = detector.detect(make_frame())
        assert isinstance(detections, list)
        for detection in detections:
            assert isinstance(detection, FaceDetection)
    finally:
        detector.close()


@requires_model("mesh")
def test_mesh_returns_face_landmarks_contract() -> None:
    mesh = MediaPipeFaceMesh(str(_MODELS["mesh"]))
    try:
        meshes = mesh.estimate(make_frame())
        assert isinstance(meshes, list)
        for landmarks in meshes:
            assert isinstance(landmarks, FaceLandmarks)
    finally:
        mesh.close()


def test_detector_invalid_confidence_raises_value_error() -> None:
    with pytest.raises(ValueError):
        MediaPipeFaceDetector(
            str(_MODELS["detector"]), min_detection_confidence=0.0
        )