"""Demo end-to-end: camera -> Vision Engine -> Feature Extractor -> AI Engine.

Reproducible single-process flow that mirrors the backend wiring: reads
frames from a camera source, runs them through the real Vision pipeline,
feeds each ``VisionEvent`` to the shared ``FeatureExtractor``, and scores
every derived ``FatigueFeatures`` with the (simulated) ``StubFatiguePredictor``.

Runs from the backend environment:

    uv run --project apps/backend python -m scripts.demo_flow [--frames N]

Deliberately does NOT use the FastAPI TestClient (whose MediaPipe workers can
hang the interpreter at exit on Windows). Requires the MediaPipe models under
``models/vision``; when missing, the script prints how to fetch them and
exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys

from ai_engine.pipeline import FatiguePipeline
from ai_engine.stub import StubFatiguePredictor
from feature_extractor import FeatureExtractor
from vision_engine.camera import CameraSettings, create_camera_source
from vision_engine.mediapipe import MediaPipeFaceDetector, MediaPipeFaceMesh
from vision_engine.pipeline import VisionPipeline

from app.config import Settings

for _stream in (sys.stdout, sys.stderr):
    try:
        # typeshed types sys streams as TextIO, which lacks reconfigure.
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

_MODELS_HINT = (
    "Los modelos de vision no estan en models/vision. Descargalos desde la raiz del repo:\n"
    "    uv run --no-project --python 3.12 scripts/fetch_models.py"
)


def _models_available(settings: Settings) -> bool:
    return settings.face_detector_model.is_file() and settings.face_landmarker_model.is_file()


def main(argv: list[str] | None = None) -> int:
    """Run the demo flow; returns the process exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "Demo E2E: cámara -> Vision Engine -> Feature Extractor -> "
            "AI Engine -> JSON (predictor simulado)."
        )
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Cantidad de frames a procesar (por defecto 30).",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    if not _models_available(settings):
        print(f"[error] {_MODELS_HINT}", file=sys.stderr)
        return 1

    camera_settings = CameraSettings(
        device_index=settings.camera_index,
        width=settings.synthetic_width,
        height=settings.synthetic_height,
        fps=settings.synthetic_fps,
    )
    camera = create_camera_source(settings.camera_source, camera_settings)
    detector = MediaPipeFaceDetector(str(settings.face_detector_model))
    mesh = MediaPipeFaceMesh(str(settings.face_landmarker_model))
    pipeline = VisionPipeline(
        detector=detector, mesh=mesh, source=settings.camera_source
    )
    extractor = FeatureExtractor()
    predictor = FatiguePipeline(StubFatiguePredictor())

    frames_with_face = 0
    features_count = 0
    try:
        for index in range(1, args.frames + 1):
            frame = camera.read()
            event = pipeline.process_frame(frame)
            if event.detections:
                frames_with_face += 1
            features = extractor.update(event)
            if features is None:
                print(f"Frame {index}/{args.frames}: sin rostro")
                continue
            result = predictor.assess(features)
            features_count += 1
            print(f"Frame {index}/{args.frames}: {json.dumps(result.to_dict())}")
    finally:
        camera.close()
        detector.close()
        mesh.close()

    print("--- Resumen ---")
    print(f"Frames procesados: {args.frames}")
    print(f"Frames con rostro detectado: {frames_with_face}")
    print(f"FatigueFeatures generadas y evaluadas: {features_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())