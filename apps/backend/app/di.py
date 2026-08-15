"""Composition root: wiring settings, adapters and the FastAPI application."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from ai_engine import HeuristicFatigueModel
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


def build_adapters(settings: Settings) -> tuple[VisionAdapter, AIAdapter]:
    """Build the production adapters from the given settings.

    Production uses the ``HeuristicFatigueModel`` predictor (S1-17): the real
    fatigue/risk evaluation. The ``StubFatiguePredictor`` remains the
    ``AIAdapter`` DEFAULT so tests and scripts that build an ``AIAdapter()``
    with no arguments keep the deterministic stub — the heuristic is wired
    ONLY here, at the composition root.
    """
    return VisionAdapter(settings), AIAdapter(predictor=HeuristicFatigueModel())


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