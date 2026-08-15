"""Factory that builds a camera source from its string kind."""

from vision_engine.camera.opencv import OpenCVCameraSource
from vision_engine.camera.settings import CameraSettings
from vision_engine.camera.synthetic import SyntheticCameraSource
from vision_engine.ports import CameraSource


def create_camera_source(
    kind: str, settings: CameraSettings | None = None
) -> CameraSource:
    """Return a camera source for the given kind: ``opencv`` or ``synthetic``."""
    if kind == "synthetic":
        return SyntheticCameraSource(settings)
    if kind == "opencv":
        return OpenCVCameraSource(settings or CameraSettings())
    raise ValueError(
        f"Unknown camera source kind: {kind!r} — expected 'opencv' or 'synthetic'"
    )