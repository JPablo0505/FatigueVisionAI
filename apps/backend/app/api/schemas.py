"""Pydantic schemas for the HTTP API.

The response contracts below are TEMPORARY: the single-shot capture shapes
will be superseded by the S1-15 realtime streaming contract. Feature values
are produced upstream by ``fatiguevision-feature-extractor`` from
``VisionEvent`` landmarks and only mirrored here so the endpoint stays
JSON-clean (no numpy, no engine dataclasses leak out).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HeadPoseModel(BaseModel):
    """Head orientation angles in radians (yaw, pitch, roll)."""

    yaw: float
    pitch: float
    roll: float


class IrisDirectionModel(BaseModel):
    """Normalized iris gaze direction in the -1..1 range."""

    x: float
    y: float


class FatigueFeaturesModel(BaseModel):
    """Mirror of ``ai_engine.FatigueFeatures`` as produced by the extractor.

    The live indicators (``current_yawn_duration``, ``last_yawn_duration``,
    ``mar_instant``) default to 0.0 so ``model_validate`` stays tolerant of
    older feature dicts that predate them (S1-16, ADITIVO). The S1-17 session
    fields (``ear_instant``, ``current_eye_closure_duration``, the longest
    durations and session totals) follow the same additive convention.
    """

    ear: float
    mar: float
    perclos: float
    blink_rate: float
    head_pose: HeadPoseModel
    eye_closure_duration: float
    yawn_duration: float
    iris_direction: IrisDirectionModel
    window_seconds: float
    current_yawn_duration: float = 0.0
    last_yawn_duration: float = 0.0
    mar_instant: float = 0.0
    ear_instant: float = 0.0
    current_eye_closure_duration: float = 0.0
    longest_eye_closure_duration: float = 0.0
    longest_yawn_duration: float = 0.0
    total_blinks: float = 0.0
    total_yawns: float = 0.0


class FatigueResultModel(BaseModel):
    """Mirror of ``ai_engine.FatigueResult.to_dict()``."""

    score: float
    level: Literal["low", "medium", "high"]
    confidence: float
    timestamp: float | None = None
    state: Literal[
        "normal",
        "long_blink",
        "fatigue_warning",
        "microsleep_suspected",
        "critical_microsleep",
    ] = "normal"


class AssessResponse(BaseModel):
    """Single-shot assessment envelope.

    TEMPORARY shape; expected to be replaced by the S1-15 streaming contract.

    ``status == "ok"`` echoes the ``features`` that were sent to the
    predictor plus the ``assessment`` produced by it. ``status ==
    "no_face"`` means the latest frame had no usable face, so no
    ``FatigueFeatures`` could be derived and nothing was submitted to the
    predictor; ``message`` carries the human-readable reason.
    """

    status: Literal["ok", "no_face"] = "ok"
    features: FatigueFeaturesModel | None = None
    assessment: FatigueResultModel | None = None
    message: str | None = None