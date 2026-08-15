"""Pure data contracts exchanged by the Vision Engine."""

from dataclasses import dataclass

import numpy as np

from vision_engine.errors import ProcessingError


@dataclass(frozen=True, slots=True)
class Frame:
    """A single video frame as a raw pixel buffer, plus capture metadata."""

    data: np.ndarray
    timestamp_ns: int

    @property
    def width(self) -> int:
        """Width in pixels, derived from the second image axis."""
        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        """Height in pixels, derived from the first image axis."""
        return int(self.data.shape[0])

    @property
    def channels(self) -> int:
        """Number of channels (1 for 2D grayscale frames)."""
        if self.data.ndim == 2:
            return 1
        if self.data.ndim == 3:
            return int(self.data.shape[2])
        raise ProcessingError(
            f"Unexpected frame dimensions {self.data.ndim}: expected 2D or 3D"
        )


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """A detected face bounding box and its confidence score."""

    bbox: tuple[int, int, int, int]
    score: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-clean representation of this detection."""
        return {"bbox": list(self.bbox), "score": self.score}


@dataclass(frozen=True, slots=True)
class FaceLandmarks:
    """Facial landmark points in normalized coordinates (x, y, z)."""

    points: tuple[tuple[float, float, float], ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-clean representation of this landmark mesh."""
        return {"points": [list(point) for point in self.points]}


@dataclass(frozen=True, slots=True)
class VisionEvent:
    """Pipeline output: metadata plus detections and meshes for one frame."""

    frame_id: int
    timestamp_ns: int
    source: str
    detections: list[FaceDetection]
    meshes: list[FaceLandmarks]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-clean dict free of numpy objects."""
        return {
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "source": self.source,
            "face_count": len(self.detections),
            "detections": [detection.to_dict() for detection in self.detections],
            "meshes": [mesh.to_dict() for mesh in self.meshes],
        }