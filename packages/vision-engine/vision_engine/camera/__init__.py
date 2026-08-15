"""Camera sources: OpenCV (real device) and synthetic (deterministic)."""

from vision_engine.camera.factory import create_camera_source
from vision_engine.camera.opencv import OpenCVCameraSource
from vision_engine.camera.settings import CameraSettings
from vision_engine.camera.synthetic import SyntheticCameraSource

__all__ = [
    "CameraSettings",
    "OpenCVCameraSource",
    "SyntheticCameraSource",
    "create_camera_source",
]