"""Vision Engine public API."""

from vision_engine.camera.factory import create_camera_source
from vision_engine.camera.opencv import OpenCVCameraSource
from vision_engine.camera.settings import CameraSettings
from vision_engine.camera.synthetic import SyntheticCameraSource
from vision_engine.drawing import draw_detections, draw_mesh
from vision_engine.errors import (
    CameraNotAvailableError,
    ProcessingError,
    VisionEngineError,
)
from vision_engine.mediapipe import MediaPipeFaceDetector, MediaPipeFaceMesh
from vision_engine.pipeline import VisionPipeline
from vision_engine.ports import (
    CameraSource,
    FaceDetector,
    FaceMeshEstimator,
    FrameProcessor,
)
from vision_engine.types import FaceDetection, FaceLandmarks, Frame, VisionEvent

__all__ = [
    "CameraNotAvailableError",
    "CameraSettings",
    "CameraSource",
    "FaceDetection",
    "FaceDetector",
    "FaceLandmarks",
    "FaceMeshEstimator",
    "Frame",
    "FrameProcessor",
    "MediaPipeFaceDetector",
    "MediaPipeFaceMesh",
    "OpenCVCameraSource",
    "ProcessingError",
    "SyntheticCameraSource",
    "VisionEngineError",
    "VisionEvent",
    "VisionPipeline",
    "create_camera_source",
    "draw_detections",
    "draw_mesh",
]