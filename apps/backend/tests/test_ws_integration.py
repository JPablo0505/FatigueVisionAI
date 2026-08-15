"""Opt-in end-to-end streaming test: real camera + pipeline through the manager.

Runs the REAL synthetic camera + MediaPipe pipeline + extractor + stub
predictor through the StreamingManager WITHOUT TestClient (which is the
pattern that hangs the interpreter on Windows at exit). Completing the
scenario is the proof that the shared loop shuts the native resources down
cleanly. Enable with:

    FATIGUE_INTEGRATION_TESTS=1 uv run pytest -q

Requires the MediaPipe models under ``models/vision``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import cast

import pytest
from ai_engine import StubFatiguePredictor
from feature_extractor import FeatureExtractor

from app.adapters.ai import AIAdapter
from app.adapters.vision import VisionAdapter
from app.config import Settings
from app.services.assessment import AssessmentService
from app.services.streaming import StreamingManager, Subscriber


def _integration_enabled() -> bool:
    return os.getenv("FATIGUE_INTEGRATION_TESTS") == "1"


def _models_present() -> bool:
    settings = Settings()
    return settings.face_detector_model.is_file() and settings.face_landmarker_model.is_file()


@pytest.mark.skipif(
    not _integration_enabled() or not _models_present(),
    reason="Needs FATIGUE_INTEGRATION_TESTS=1 and the MediaPipe models downloaded",
)
def test_realtime_stream_with_real_pipeline_shuts_down_cleanly() -> None:
    settings = Settings()
    vision = VisionAdapter(settings)  # real camera + MediaPipe pipeline, built lazily
    ai = AIAdapter(StubFatiguePredictor())
    service = AssessmentService(vision, ai, FeatureExtractor())
    manager = StreamingManager(vision, service)

    async def scenario() -> tuple[int, int]:
        sub = Subscriber()
        await manager.subscribe(sub)
        await manager.start()

        statuses = 0
        assessments = 0
        errors = 0
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                text = await asyncio.wait_for(sub.queue.get(), timeout=0.2)
            except TimeoutError:
                continue
            message = cast(dict[str, object], json.loads(text))
            kind = message["kind"]
            if kind == "vision.status":
                statuses += 1
            elif kind == "fatigue.assessment":
                assessments += 1
            elif kind == "error":
                errors += 1
        await manager.shutdown()
        return statuses, errors

    statuses, errors = asyncio.run(scenario())

    assert statuses >= 1, "the periodic status tick must flow on real frames"
    assert errors == 0, f"real pipeline reported errors: {errors}"
    # The shutdown dropped the streaming state and released the camera.
    assert manager.state == "stopped"
    status = vision.status()
    camera = cast(dict[str, object], status["camera"])
    assert camera["running"] is False