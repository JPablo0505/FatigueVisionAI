"""FatigueVision AI backend entry point (uvicorn targets ``app.main:app``)."""

from app.di import create_app

app = create_app()