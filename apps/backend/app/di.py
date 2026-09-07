"""Composition root: wiring settings, adapters and the FastAPI application."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI
from feature_extractor import FeatureExtractor

from app.adapters.ai import AIAdapter
from app.adapters.vision import VisionAdapter
from app.api.routes import assess, health, ws
from app.api.routes import vision as vision_router
from app.config import Settings
from app.services.assessment import AssessmentService
from app.services.streaming import StreamingManager

APP_NAME = os.getenv("APP_NAME", "FatigueVision AI")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """App lifecycle gate.

    On shutdown the streaming manager cancels its capture loop and status
    ticker cooperatively and releases the camera the whole app shares.
    """
    try:
        yield
    finally:
        manager = cast(StreamingManager, app.state.streaming_manager)
        await manager.shutdown()


def _build_fatigue_model(settings: Settings):
    """Instantiate the fatigue model based on the feature flag.

    ``"ml"``        → ``MLFatigueModel`` (trained RandomForest, fatigue_v2.joblib)
    ``"heuristic"`` → ``HeuristicFatigueModel`` (rule-based, S1-17)

    The ``FatigueModel`` port is shared: both implementations produce a
    ``FatigueResult`` with the same contract, so the rest of the pipeline
    is untouched.
    """
    model_type = settings.fatigue_model_type

    if model_type == "ml":
        from ai_engine.ml_model import MLFatigueModel

        logger.info("Fatigue model: ML (fatigue_v2.joblib)")
        return MLFatigueModel()

    if model_type == "heuristic":
        from ai_engine import HeuristicFatigueModel

        logger.info("Fatigue model: Heuristic (rule-based)")
        return HeuristicFatigueModel()

    raise ValueError(
        f"Unknown FATIGUE_MODEL_TYPE={model_type!r}. "
        "Expected 'ml' or 'heuristic'."
    )


def build_adapters(settings: Settings) -> tuple[VisionAdapter, AIAdapter]:
    """Build the production adapters from the given settings.

    The fatigue model is selected by ``Settings.fatigue_model_type``
    (env var ``FATIGUE_MODEL_TYPE``). Default is ``"ml"`` (S3-01).
    """
    return VisionAdapter(settings), AIAdapter(predictor=_build_fatigue_model(settings))


def build_extractor(settings: Settings) -> FeatureExtractor:
    """Build the SINGLETON ``FeatureExtractor`` used by the whole application.

    Deliberately constructed ONCE per app instance (not per request): its
    sliding window must accumulate landmarks across calls so PERCLOS, blink
    rate and episode durations have temporal context. A per-request
    extractor would reset the window on every call and never produce
    meaningful aggregates. The default 30-second configuration comes from
    ``FeatureExtractorConfig``.
    """
    return FeatureExtractor()


def create_app(
    settings: Settings | None = None,
    *,
    vision: VisionAdapter | None = None,
    ai: AIAdapter | None = None,
    extractor: FeatureExtractor | None = None,
) -> FastAPI:
    """Build the application; injected adapters win over built defaults."""
    settings = settings or Settings()
    if vision is None or ai is None:
        default_vision, default_ai = build_adapters(settings)
    vision = vision or default_vision
    ai = ai or default_ai
    extractor = extractor or build_extractor(settings)
    service = AssessmentService(vision, ai, extractor)
    streaming = StreamingManager(vision, service)

    app = FastAPI(title=f"{APP_NAME} API", version="0.1.0", lifespan=_lifespan)
    app.state.assessment_service = service
    app.state.streaming_manager = streaming
    app.include_router(health.router)
    app.include_router(vision_router.router)
    app.include_router(assess.router)
    app.include_router(ws.router)
    return app