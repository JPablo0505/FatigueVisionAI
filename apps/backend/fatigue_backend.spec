# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for FatigueVision AI backend.

Bundles the FastAPI server, AI engine, feature extractor, vision engine,
MediaPipe models, and the fatigue_v3.joblib ML model into a single
portable directory executable.

Usage (from repo root):
    pyinstaller apps/backend/fatigue_backend.spec
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPEC_DIR = Path(SPECPATH)  # noqa: F821 — injected by PyInstaller
REPO_ROOT = SPEC_DIR.parent.parent.parent
BACKEND_DIR = REPO_ROOT / "apps" / "backend"
PACKAGES_DIR = REPO_ROOT / "packages"
MODELS_DIR = REPO_ROOT / "models"
AI_MODELS_DIR = PACKAGES_DIR / "ai-engine" / "ai_engine" / "models"
FRONTEND_DIST = REPO_ROOT / "apps" / "web" / "dist"

# ---------------------------------------------------------------------------
# Data files to bundle (src, dest)
# ---------------------------------------------------------------------------
datas = [
    # MediaPipe vision models
    (str(MODELS_DIR / "vision" / "face_detector.tflite"), "models/vision"),
    (str(MODELS_DIR / "vision" / "face_landmarker.task"), "models/vision"),
    # Fatigue ML model
    (str(AI_MODELS_DIR / "fatigue_v3.joblib"), "ai_engine/models"),
    # Frontend static files (built by Vite)
    (str(FRONTEND_DIST), "static/web"),
]

# ---------------------------------------------------------------------------
# Hidden imports — packages loaded dynamically or via entry points
# ---------------------------------------------------------------------------
hiddenimports = [
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "fastapi",
    "pydantic",
    "pydantic_settings",
    "ai_engine",
    "ai_engine.ml_model",
    "ai_engine.heuristic",
    "ai_engine.pipeline",
    "ai_engine.types",
    "ai_engine.ports",
    "ai_engine.errors",
    "ai_engine.stub",
    "feature_extractor",
    "vision_engine",
    "vision_engine.mediapipe",
    "vision_engine.camera",
    "vision_engine.pipeline",
    "vision_engine.types",
    "vision_engine.ports",
    "vision_engine.errors",
    "app",
    "app.main",
    "app.config",
    "app.di",
    "app.adapters",
    "app.adapters.ai",
    "app.adapters.vision",
    "app.adapters.browser_camera",
    "app.api",
    "app.api.routes",
    "app.api.routes.health",
    "app.api.routes.assess",
    "app.api.routes.ws",
    "app.api.routes.vision",
    "app.services",
    "app.services.assessment",
    "app.services.streaming",
    "app.protocol",
    "starlette",
    "starlette.routing",
    "starlette.responses",
    "starlette.websockets",
    "httpx",
    "websockets",
    "numpy",
    "joblib",
    "sklearn",
    "sklearn.ensemble",
    "sklearn.utils._typedefs",
    "sklearn.neighbors._partition_nodes",
    "scipy",
    "scipy.spatial",
    "scipy.special",
]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    [str(BACKEND_DIR / "app" / "main.py")],
    pathex=[str(BACKEND_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "PIL",
        "pytest",
        "mypy",
        "ruff",
    ],
    noarchive=False,
    optimize=2,
)

# ---------------------------------------------------------------------------
# PYZ — compressed Python archive
# ---------------------------------------------------------------------------
pyz = PYZ(a.pure, a.zipped_data)  # noqa: F821

# ---------------------------------------------------------------------------
# EXE — the entry point
# ---------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="fatigue_backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

# ---------------------------------------------------------------------------
# COLLECT — output directory with all files
# ---------------------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="fatigue_backend",
)
