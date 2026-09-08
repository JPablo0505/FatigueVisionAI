"""FatigueVision AI backend entry point (uvicorn targets ``app.main:app``)."""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from app.config import Settings  # noqa: E402
from app.di import create_app  # noqa: E402

_settings = Settings()
logger.info(
    "Startup config: CAMERA_SOURCE=%s  FATIGUE_MODEL_TYPE=%s",
    _settings.camera_source,
    _settings.fatigue_model_type,
)

app = create_app(_settings)