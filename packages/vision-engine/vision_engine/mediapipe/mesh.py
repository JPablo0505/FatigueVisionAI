"""MediaPipe Tasks face landmarker implementing the ``FaceMeshEstimator`` port.

Uses the modern Tasks API (``mediapipe.tasks.python.vision.FaceLandmarker``),
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
from vision_engine.types import FaceLandmarks, Frame


class MediaPipeFaceMesh:
    """Adapts ``mediapipe.tasks.python.vision.FaceLandmarker`` to the port.

    Implemented structurally against the ``FaceMeshEstimator`` protocol: it
    exposes ``estimate`` without inheriting from the Protocol. ``close`` is
    not part of the port but is provided so the orchestrator can release the
    underlying native task.
    """

    def __init__(
        self,
        model_path: str,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Load the Face Landmarker model into a single-image MediaPipe task.

        ``min_detection_confidence`` and ``min_tracking_confidence`` follow
        the port's naming; they map onto the Tasks API's
        ``min_face_detection_confidence`` and ``min_tracking_confidence``.
        Raises ``ValueError`` if the face count is not at least 1 and
        ``ProcessingError`` if the model file cannot be loaded.
        """
        if max_num_faces < 1:
            raise ValueError(f"max_num_faces must be >= 1, got {max_num_faces}")
        if not 0 < min_detection_confidence <= 1:
            raise ValueError(
                "min_detection_confidence must be in (0, 1], "
                f"got {min_detection_confidence}"
            )
        if not 0 < min_tracking_confidence <= 1:
            raise ValueError(
                "min_tracking_confidence must be in (0, 1], "
                f"got {min_tracking_confidence}"
            )
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        try:
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            raise ProcessingError(
                f"Cannot load MediaPipe face landmarker model at {model_path}: {exc}"
            ) from exc
        self._closed = False

    def estimate(self, frame: Frame) -> list[FaceLandmarks]:
        """Return the landmark meshes estimated for the frame as port objects.

        The BGR input is converted to RGB here; landmark points are emitted
        normalized (x, y, z in 0..1) exactly as the port contract requires.
        """
        rgb = np.ascontiguousarray(frame.data[:, :, ::-1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(image)
        meshes: list[FaceLandmarks] = []
        for mesh in results.face_landmarks:
            points = tuple(
                (float(landmark.x), float(landmark.y), float(landmark.z))
                for landmark in mesh
            )
            meshes.append(FaceLandmarks(points=points))
        return meshes

    def close(self) -> None:
        """Release the underlying MediaPipe task; safe to call twice."""
        if self._closed:
            return
        self._landmarker.close()
        self._closed = True