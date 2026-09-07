"""DDD (Driver Drowsiness Dataset) adapter.

Reads DDD PNG image files and produces ``VisionEvent`` objects with
real face detections and 478-point landmark meshes via MediaPipe.

DDD structure (from Kaggle):
    Two-class image dataset:
        Drowsy/      — 22,348 PNG images (uppercase letter prefixes A-Z)
        Non Drowsy/  — 19,445 PNG images (lowercase letter prefixes a-z)

    Naming convention: ``{letter}{number}.png``
        - Letter prefix identifies the subject.
        - Uppercase = drowsy class, lowercase = non-drowsy class.
        - Number is a frame index within the subject's session.

This adapter:
    1. Reads PNG images via OpenCV/PIL.
    2. Runs MediaPipe face detection + mesh estimation on each image.
    3. Yields ``(VisionEvent, labels)`` pairs for the ``DatasetProcessor``.
    4. Derives per-image labels from the directory membership and filename.

Limitations documented in ``datasets/README.md``:
    - Labels are binary (drowsy/non_drowsy), not per-frame fatigue levels.
    - DDD does not contain fine-grained fatigue annotations.
    - EAR, MAR, PERCLOS etc. are computed by the FeatureExtractor pipeline,
      NOT by this adapter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from vision_engine.types import FaceDetection, FaceLandmarks, Frame, VisionEvent

from dataset_manager.types import DatasetInfo, DatasetPort

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDD filename convention parsing
# ---------------------------------------------------------------------------

# Pattern: {letter}{number}.png
# Letter case determines class: uppercase = Drowsy, lowercase = Non Drowsy
_DDD_FILENAME_RE = None  # We parse manually since the pattern is simple


def _parse_filename(filename: str) -> dict[str, object] | None:
    """Extract metadata from a DDD filename.

    Handles both Drowsy (uppercase) and Non Drowsy (lowercase) naming.

    Returns a dict with subject_id, fatigue_level, and frame_number,
    or None if the filename doesn't match the expected convention.
    """
    # Normalize: strip whitespace, remove .png extension
    normalized = filename.strip()
    if normalized.lower().endswith(".png"):
        normalized = normalized[:-4]

    if not normalized:
        return None

    # First character is the subject letter
    letter = normalized[0]
    if not letter.isalpha():
        return None

    # Remaining characters should be digits
    digits = normalized[1:]
    if not digits.isdigit():
        return None

    # Class from letter case
    fatigue_level = "drowsy" if letter.isupper() else "non_drowsy"
    subject_id = letter.lower()  # Normalize to lowercase for consistency

    return {
        "subject_id": subject_id,
        "fatigue_level": fatigue_level,
        "frame_number": int(digits),
    }


def _list_image_files(root: Path) -> list[tuple[Path, str]]:
    """Discover PNG files under the DDD directory structure.

    Returns list of (path, class_label) tuples.
    """
    images: list[tuple[Path, str]] = []

    # Drowsy class
    drowsy_dir = root / "Drowsy"
    if drowsy_dir.is_dir():
        for img_path in sorted(drowsy_dir.glob("*.png")):
            images.append((img_path, "drowsy"))

    # Non Drowsy class
    non_drowsy_dir = root / "Non Drowsy"
    if non_drowsy_dir.is_dir():
        for img_path in sorted(non_drowsy_dir.glob("*.png")):
            images.append((img_path, "non_drowsy"))

    return images


@dataclass(frozen=True, slots=True)
class DDDConfig:
    """Configuration for the DDD adapter.

    Attributes:
        ddd_root:     Root directory of the DDD dataset (contains
                      ``Drowsy/`` and ``Non Drowsy/`` subdirectories).
        face_detector:  Path to the MediaPipe face detector model.
        face_landmarker: Path to the MediaPipe face landmarker model.
        min_detection_confidence: Minimum face detection confidence.
        max_faces:      Maximum faces to detect per image (1 = single-subject).
        image_skip:     Process every Nth image (1 = all, 2 = every other).
        synthetic_fps:  FPS to assume for timestamp generation (since images
                        are independent, this controls the timestamp spacing).
    """

    ddd_root: Path
    face_detector: Path = Path("models/vision/face_detector.tflite")
    face_landmarker: Path = Path("models/vision/face_landmarker.task")
    min_detection_confidence: float = 0.5
    max_faces: int = 1
    image_skip: int = 1
    synthetic_fps: float = 30.0


class DDDAdapter(DatasetPort):
    """Adapter that reads DDD PNG images and produces VisionEvents.

    Each image is read via OpenCV. Face detection and 478-point landmark
    mesh estimation run via MediaPipe on every (or every Nth) image.
    The adapter yields complete ``VisionEvent`` objects that the
    ``FeatureExtractor`` can consume directly.

    Usage::

        config = DDDConfig(ddd_root=Path("datasets/raw/Driver Drowsiness Dataset (DDD)"))
        adapter = DDDAdapter(config)
        errors = adapter.validate()
        for event, labels in adapter.iter_events():
            ...
    """

    def __init__(self, config: DDDConfig | None = None) -> None:
        if config is None:
            raise ValueError(
                "DDDAdapter requires an explicit DDDConfig with ddd_root"
            )
        self._config = config
        self._detector = None
        self._mesh_estimator = None

    def _ensure_models(self) -> None:
        """Lazy-load MediaPipe models on first use."""
        if self._detector is not None:
            return

        from vision_engine.mediapipe import MediaPipeFaceDetector, MediaPipeFaceMesh

        detector_path = str(self._config.face_detector)
        mesh_path = str(self._config.face_landmarker)

        logger.info("Loading face detector from %s", detector_path)
        self._detector = MediaPipeFaceDetector(
            detector_path,
            min_detection_confidence=self._config.min_detection_confidence,
        )

        logger.info("Loading face landmarker from %s", mesh_path)
        self._mesh_estimator = MediaPipeFaceMesh(
            mesh_path,
            max_num_faces=self._config.max_faces,
            min_detection_confidence=self._config.min_detection_confidence,
        )

    def info(self) -> DatasetInfo:
        image_files = _list_image_files(self._config.ddd_root)

        # Count by class
        drowsy_count = sum(1 for _, label in image_files if label == "drowsy")
        non_drowsy_count = sum(
            1 for _, label in image_files if label == "non_drowsy"
        )

        # Count unique subjects
        subjects = set()
        for path, _ in image_files:
            meta = _parse_filename(path.stem)
            if meta is not None:
                subjects.add(meta["subject_id"])

        return DatasetInfo(
            name="ddd",
            version="1.0",
            source_path=self._config.ddd_root,
            description=(
                "DDD: Driver Drowsiness Dataset. Contains PNG face images "
                "of drivers in two states: Drowsy and Non Drowsy. Images "
                "are 227x227 RGB, captured from in-car monitoring cameras."
            ),
            license="Academic use only (see Kaggle dataset terms)",
            citation=(
                "DDD: Driver Drowsiness Dataset. "
                "Available on Kaggle."
            ),
            subjects=len(subjects),
            total_frames=len(image_files),
            annotation_schema={
                "fatigue_level": {
                    "type": "categorical",
                    "values": ["drowsy", "non_drowsy"],
                    "description": (
                        "Binary drowsiness label from directory membership. "
                        "NOT hand-annotated per-frame."
                    ),
                },
                "subject_id": {"type": "string"},
                "frame_number": {"type": "integer"},
            },
            label_vocabulary={
                "fatigue_level": ["drowsy", "non_drowsy"],
            },
        )

    def validate(self) -> list[str]:
        """Validate the DDD directory structure and image files."""
        errors: list[str] = []
        root = self._config.ddd_root

        if not root.is_dir():
            errors.append(f"DDD root directory does not exist: {root}")
            return errors

        image_files = _list_image_files(root)
        if not image_files:
            errors.append(
                f"No PNG files found in {root}. "
                "Expected Drowsy/ and Non Drowsy/ subdirectories with .png files."
            )
            return errors

        # Validate a sample of files can be opened
        sample_size = min(5, len(image_files))
        for img_path, _ in image_files[:sample_size]:
            img = cv2.imread(str(img_path))
            if img is None:
                errors.append(f"Cannot read image file: {img_path.name}")
            else:
                h, w = img.shape[:2]
                if (w, h) != (227, 227):
                    errors.append(
                        f"Unexpected resolution {w}x{h} in {img_path.name} "
                        "(expected 227x227)"
                    )

        return errors

    def iter_events(self) -> Iterator[tuple[VisionEvent, dict[str, object]]]:
        """Yield (VisionEvent, labels) pairs from DDD images."""
        self._ensure_models()
        assert self._detector is not None
        assert self._mesh_estimator is not None

        image_files = _list_image_files(self._config.ddd_root)
        logger.info(
            "DDD adapter: %d image files to process",
            len(image_files),
        )

        frame_counter = 0
        fps = self._config.synthetic_fps

        for img_path, class_label in image_files:
            # Parse filename metadata
            meta = _parse_filename(img_path.stem)
            if meta is None:
                logger.warning(
                    "Skipping %s: filename doesn't match DDD convention",
                    img_path.name,
                )
                continue

            # Image skip
            if frame_counter % self._config.image_skip != 0:
                frame_counter += 1
                continue

            # Read image
            bgr_frame = cv2.imread(str(img_path))
            if bgr_frame is None:
                logger.warning("Cannot read %s, skipping", img_path.name)
                continue

            # Build Frame object for MediaPipe
            timestamp_ns = int((frame_counter / fps) * 1_000_000_000)
            frame = Frame(data=bgr_frame, timestamp_ns=timestamp_ns)

            # Run face detection + mesh estimation
            detections = self._detector.detect(frame)
            meshes = self._mesh_estimator.estimate(frame)

            # Base labels for this image
            base_labels: dict[str, object] = {
                "subject_id": meta["subject_id"],
                "fatigue_level": meta["fatigue_level"],
                "frame_number": meta["frame_number"],
                "image_file": img_path.name,
                "class_dir": class_label,
            }

            # Per-image labels
            frame_labels = {
                **base_labels,
                "clip_id": img_path.stem,
                "frame_idx": frame_counter,
                "fps": fps,
            }

            event = VisionEvent(
                frame_id=frame_counter,
                timestamp_ns=timestamp_ns,
                source="ddd",
                detections=detections,
                meshes=meshes,
            )

            frame_counter += 1
            yield event, frame_labels

        logger.info(
            "DDD adapter: yielded %d images total", frame_counter
        )

    def close(self) -> None:
        """Release MediaPipe models; safe to call multiple times."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        if self._mesh_estimator is not None:
            self._mesh_estimator.close()
            self._mesh_estimator = None
