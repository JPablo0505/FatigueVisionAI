"""Opt-in end-to-end integration test: camera -> pipeline -> extractor -> predictor.

Runs the real MediaPipe pipeline DIRECTLY (no TestClient, which is the
pattern that hangs the interpreter on Windows at exit). Enable with:

    FATIGUE_INTEGRATION_TESTS=1 uv run pytest -q

Requires the MediaPipe models under ``models/vision`` (see
``scripts/fetch_models.py`` at the repo root).
"""

from __future__ import annotations

import json
import os

import pytest
from ai_engine.pipeline import FatiguePipeline
from ai_engine.stub import StubFatiguePredictor
from feature_extractor import FeatureExtractor
from vision_engine.camera import CameraSettings, create_camera_source
from vision_engine.mediapipe import MediaPipeFaceDetector, MediaPipeFaceMesh
from vision_engine.pipeline import VisionPipeline

from app.config import Settings


def _integration_enabled() -> bool:
    return os.getenv("FATIGUE_INTEGRATION_TESTS") == "1"


def _models_present() -> bool:
    settings = Settings()
    return settings.face_detector_model.is_file() and settings.face_landmarker_model.is_file()


@pytest.mark.skipif(
    not _integration_enabled() or not _models_present(),
    reason="Needs FATIGUE_INTEGRATION_TESTS=1 and the MediaPipe models downloaded",
)
def test_e2e_flow_produces_valid_json_and_no_crash() -> None:
    settings = Settings()
    camera = create_camera_source(
        "synthetic",
        CameraSettings(
            device_index=settings.camera_index,
            width=settings.synthetic_width,
            height=settings.synthetic_height,
            fps=settings.synthetic_fps,
        ),
    )
    detector = MediaPipeFaceDetector(str(settings.face_detector_model))
    mesh = MediaPipeFaceMesh(str(settings.face_landmarker_model))
    pipeline = VisionPipeline(detector=detector, mesh=mesh, source="synthetic")
    extractor = FeatureExtractor()
    predictor = FatiguePipeline(StubFatiguePredictor())

    frames = 10
    frames_with_face = 0
    features_count = 0
    results: list[dict[str, object]] = []
    try:
        for _ in range(frames):
            frame = camera.read()
            event = pipeline.process_frame(frame)
            if event.detections:
                frames_with_face += 1
            features = extractor.update(event)
            if features is None:
                continue
            features_count += 1
            result = predictor.assess(features).to_dict()
            assert set(result) == {"score", "level", "confidence", "timestamp", "state"}
            assert isinstance(json.dumps(result), str)
            results.append(result)
    finally:
        camera.close()
        detector.close()
        mesh.close()

    assert frames_with_face >= 0
    assert features_count == len(results)
    assert extractor.window_seconds == 30.0