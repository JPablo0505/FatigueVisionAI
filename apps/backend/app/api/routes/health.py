"""Health and service identity endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/api")
def api_info() -> dict[str, str]:
    """Return service identity information."""
    return {"service": "fatiguevision-backend", "version": "0.1.0"}


@router.get("/api/healthz")
def healthz() -> dict[str, str]:
    """Return the service health status."""
    return {"status": "ok"}