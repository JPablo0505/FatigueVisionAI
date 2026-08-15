"""MediaPipe Tasks face detector implementing the ``FaceDetector`` port.

Uses the modern Tasks API (``mediapipe.tasks.python.vision.FaceDetector``),
not the legacy ``mediapipe.solutions`` API. Frames arrive in BGR (the OpenCV
convention) and are converted to RGB internally before being handed to
MediaPipe. Contains no business logic beyond that conversion and the mapping
of raw results onto the port types.
"""

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from vision_engine.errors import ProcessingError
from vision_engine.types import FaceDetection, Frame


class MediaPipeFaceDetector:
    """Adapts ``mediapipe.tasks.python.vision.FaceDetector`` to the port.

    Implemented structurally against the ``FaceDetector`` protocol: it
    exposes ``detect`` without inheriting from the Protocol. ``close`` is not
    part of the port but is provided so the orchestrator can release the
    underlying native task.
    """

    def __init__(
        self,
        model_path: str,
        min_detection_confidence: float = 0.5,
    ) -> None:
        """Load the BlazeFace model into a single-image MediaPipe task.

        Raises ``ValueError`` if the confidence is not in the open interval
        (0, 1] and ``ProcessingError`` if the model file cannot be loaded.
        """
        if not 0 < min_detection_confidence <= 1:
            raise ValueError(
                "min_detection_confidence must be in (0, 1], "
                f"got {min_detection_confidence}"
            )
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence,
        )
        try:
            self._detector = mp_vision.FaceDetector.create_from_options(options)
        except Exception as exc:
            raise ProcessingError(
                f"Cannot load MediaPipe face detector model at {model_path}: {exc}"
            ) from exc
        self._closed = False

    def detect(self, frame: Frame) -> list[FaceDetection]:
        """Return the faces detected in the frame as port ``FaceDetection``.

        The BGR input is converted to RGB here; detection scores come from
        the first (most confident) category, defaulting to 0.0 if the
        MediaPipe result carries no categories.
        """
        rgb = np.ascontiguousarray(frame.data[:, :, ::-1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._detector.detect(image)
        detections: list[FaceDetection] = []
        for detection in results.detections:
            bounding_box = detection.bounding_box
            score = (
                float(detection.categories[0].score)
                if detection.categories
                else 0.0
            )
            detections.append(
                FaceDetection(
                    bbox=(
                        int(bounding_box.origin_x),
                        int(bounding_box.origin_y),
                        int(bounding_box.width),
                        int(bounding_box.height),
                    ),
                    score=score,
                )
            )
        return detections

    def close(self) -> None:
        """Release the underlying MediaPipe task; safe to call twice."""
        if self._closed:
            return
        self._detector.close()
        self._closed = True