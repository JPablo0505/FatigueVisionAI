"""Ports (interfaces) the Vision Engine exposes for external adapters."""

from typing import Protocol, runtime_checkable

from vision_engine.types import FaceDetection, FaceLandmarks, Frame


@runtime_checkable
class CameraSource(Protocol):
    """Provides raw frames from a camera or any frame-producing device."""

    def read(self) -> Frame:
        """Return the next available frame, or raise CameraNotAvailableError."""
        ...

    def close(self) -> None:
        """Release any underlying device resource."""
        ...

    @property
    def closed(self) -> bool:
        """Return whether this source has been closed."""
        ...


@runtime_checkable
class FaceDetector(Protocol):
    """Detects faces in a frame and returns their bounding boxes."""

    def detect(self, frame: Frame) -> list[FaceDetection]:
        """Return the faces detected in the given frame."""
        ...


@runtime_checkable
class FaceMeshEstimator(Protocol):
    """Estimates facial landmark meshes in a frame."""

    def estimate(self, frame: Frame) -> list[FaceLandmarks]:
        """Return the landmark meshes estimated for the given frame."""
        ...


@runtime_checkable
class FrameProcessor(Protocol):
    """Applies a transformation to a frame before detection."""

    def process(self, frame: Frame) -> Frame:
        """Return the processed frame."""
        ...