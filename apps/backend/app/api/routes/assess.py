"""Fatigue assessment endpoint (single shot, camera-frame-driven)."""

from typing import Annotated, cast

from ai_engine.errors import InferenceError
from fastapi import APIRouter, Depends, HTTPException, Request
from vision_engine.errors import CameraNotAvailableError, ProcessingError

from app.api.schemas import AssessResponse
from app.services.assessment import AssessmentService

router = APIRouter(prefix="/api", tags=["assess"])


def _get_service(request: Request) -> AssessmentService:
    return cast(AssessmentService, request.app.state.assessment_service)


@router.post("/assess", response_model=AssessResponse)
def assess(
    service: Annotated[AssessmentService, Depends(_get_service)],
) -> AssessResponse:
    """Capture one frame and score fatigue over the last 30 seconds.

    The backend captures a single frame from the configured camera, runs it
    through the Vision pipeline, feeds the resulting ``VisionEvent`` to the
    shared ``FeatureExtractor`` and sends the derived ``FatigueFeatures`` to
    the AI engine. A frame without a usable face yields the ``no_face``
    envelope instead (HTTP 200; it is a valid observation, not an error).
    """
    try:
        return service.assess()
    except CameraNotAvailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "camera.unavailable", "message": str(exc)},
        ) from exc
    except ProcessingError as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "pipeline.error", "message": str(exc)},
        ) from exc
    except InferenceError as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "inference.error", "message": str(exc)},
        ) from exc