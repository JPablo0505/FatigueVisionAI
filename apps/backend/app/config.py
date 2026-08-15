"""Application settings loaded from environment variables.

Backend is the orchestrator: all values here only select which engine
resources to build, never what they compute.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "vision"


class Settings(BaseSettings):
    """Runtime configuration. Fields are mapped from UPPERCASE env vars."""

    # "browser" = frames arrive over the WebSocket (frame.provide); the backend
    # opens no device in that mode.
    camera_source: Literal["opencv", "synthetic", "browser"] = "synthetic"
    camera_index: int = 0
    synthetic_width: int = 640
    synthetic_height: int = 480
    synthetic_fps: int = 30

    # MediaPipe models live under REPO_ROOT/models/vision.
    model_dir: Path = DEFAULT_MODEL_DIR
    face_detector_model: Path = DEFAULT_MODEL_DIR / "face_detector.tflite"
    face_landmarker_model: Path = DEFAULT_MODEL_DIR / "face_landmarker.task"