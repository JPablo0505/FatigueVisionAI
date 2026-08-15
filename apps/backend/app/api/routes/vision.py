"""Vision status and one-shot frame capture endpoints."""

from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from vision_engine.errors import CameraNotAvailableError, ProcessingError

from app.services.assessment import AssessmentService

router = APIRouter(prefix="/api/vision", tags=["vision"])


def _get_service(request: Request) -> AssessmentService:
    return cast(AssessmentService, request.app.state.assessment_service)


@router.get("/status")
def vision_status(
    service: Annotated[AssessmentService, Depends(_get_service)],
) -> dict[str, object]:
    """Return the camera/pipeline status (``VisionStatus`` shape)."""
    return service.vision_status()


@router.post("/frame")
def capture_frame(
    service: Annotated[AssessmentService, Depends(_get_service)],
) -> dict[str, object]:
    """Capture and process a single frame (``VisionEvent`` shape)."""
    try:
        return service.capture_frame()
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