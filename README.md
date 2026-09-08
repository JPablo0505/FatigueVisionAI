<p align="center">
  <img src="apps/web/src/assets/logo.png" alt="FatigueVision AI Logo" width="120" />
</p>

<h1 align="center">FatigueVision AI</h1>

<p align="center">
  <strong>Real-time driver fatigue monitoring system using computer vision and machine learning.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/Vite-6-646CFF?logo=vite&logoColor=white" alt="Vite" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-4-06B6D4?logo=tailwindcss&logoColor=white" alt="Tailwind CSS" />
  <img src="https://img.shields.io/badge/MediaPipe-Face_Mesh-FF6F00" alt="MediaPipe" />
  <img src="https://img.shields.io/badge/Scikit--Learn-RandomForest-DB2D20?logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-S4_Complete-22c55e" alt="Status" />
  <img src="https://img.shields.io/github/license/Gentleman-Programming/FatigueVisionAI" alt="License" />
</p>

<p align="center">
  <a href="README.es.md">🇪🇸 Español</a> | <strong>🇬🇧 English</strong>
</p>

---

## Key Features

- **Real-time detection** — EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio), PERCLOS, and blink rate calculated over 5-second sliding windows.
- **Optimized ML model** — `fatigue_v3.joblib` (RandomForest, CV F1-macro 0.8857) with real-time override for instant eye closures.
- **Ebony-clay dashboard** — Dark-themed control panel with risk gauge, live metrics, and facial overlay with 468 landmarks.
- **Critical alerts** — 4 types of audible alarms (beep, siren, emergency, buzzer) via Web Audio API.
- **Local processing** — The entire pipeline runs on your machine with no external service dependencies.

## Monorepo Architecture

```
FatigueVisionAI/
├── apps/
│   ├── web/              # React + Vite + Tailwind CSS frontend
│   └── backend/          # FastAPI + WebSocket API
├── packages/
│   ├── ai-engine/        # ML model and inference pipeline
│   ├── feature-extractor/# Facial geometry and sliding windows
│   ├── vision-engine/    # Face detection with MediaPipe
│   └── contracts/        # Shared TS/Python schemas
├── datasets/             # Training data (YAWDD, etc.)
└── models/               # AI models (not committed)
```

## Local Setup Instructions

### Quick Start (Recommended)

The scripts automate dependency installation, backend and frontend startup, and open the browser automatically.

**Windows:**
```bat
start_app.bat
```

**Linux / macOS:**
```bash
bash start_app.sh
```

> The scripts detect if dependencies are already installed and skip them on future runs.

### Manual Setup

#### Prerequisites

- **Node.js** 22+ and **pnpm** (via corepack)
- **Python** 3.12+ and **uv**

#### Frontend

```bash
cd apps/web
npm install
npm run dev          # http://localhost:5173
```

#### Backend

```bash
cd apps/backend
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

uv pip install -e ../../packages/feature-extractor
uv pip install -e ../../packages/ai-engine
uv pip install -e ../../packages/vision-engine
uv pip install -e .

CAMERA_SOURCE=browser FATIGUE_MODEL_TYPE=ml uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The backend exposes `http://localhost:8000/api/vision/status` and the WebSocket at `ws://localhost:8000/ws`.

### Available Commands

| Command | Action |
|---------|--------|
| `start_app.bat` / `bash start_app.sh` | Full automatic startup |
| `npm run dev` | Frontend dev server (port 5173) |
| `npm run build` | Production build |
| `uvicorn app.main:app --reload` | Backend with hot-reload (port 8000) |

## Project Status

| Sprint | Status | Description |
|--------|--------|-------------|
| **S1** | Complete | Vision Engine, Feature Extractor, AI Engine (heuristic), Backend WS, Frontend dashboard |
| **S2** | Complete | Dataset infrastructure, trained ML model (v3, F1-macro 0.8857) |
| **S3** | Complete | Critical threshold tuned, real-time override, head-pose guard |
| **S4** | Complete | Ebony-clay UX, logo, settings panel, alarms, local distribution scripts |

**Project completed.** End-to-end functional system: browser camera → WebSocket → MediaPipe face detection → ML fatigue assessment → real-time dashboard.

---

<p align="center">
  <sub>Built with passion for road safety.</sub>
</p>
