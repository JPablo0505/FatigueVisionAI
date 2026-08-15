# FatigueVision AI

Plataforma de monitoreo de fatiga mediante visión artificial. Detecta signos de fatiga y microsueños a partir de una cámara convencional, ejecutando todo el procesamiento de forma local, sin depender de servicios externos.

Este repositorio es un monorepo que reutiliza código entre la aplicación web y la aplicación de escritorio.

> **Estado actual:** Sprint 1 completo. El sistema detecta fatiga y microsueños en tiempo real: cámara + Vision Engine (MediaPipe), Feature Extractor (ventana deslizante de 30 s), heurística de riesgo (S1-17) y dashboard web con overlay facial, eventos biométricos, alertas y sonido (S1-19). La base de datos PostgreSQL queda para el Sprint 2.

## Requisitos

| Herramienta | Versión | Obligatoria | Notas |
|-------------|---------|-------------|-------|
| Node.js | 24+ | Sí | Para web y desktop |
| pnpm | 11+ | Sí | Gestor de paquetes del workspace |
| uv | reciente | Sí | Gestor de dependencias y entornos Python |
| Docker | 29+ | Opcional | Solo para levantar el backend en contenedor |
| Rust + MSVC | — | Solo desktop | Únicamente para construir la app de escritorio |

## Estructura

```
fatiguevision-ai/
├─ apps/
│  ├─ web/        # Frontend React + Vite + Tailwind
│  ├─ backend/    # API FastAPI
│  └─ desktop/    # App de escritorio Tauri (reutiliza apps/web)
├─ packages/      # Módulos compartidos (contracts: contratos de comunicación, ui, vision-engine, ai-engine, feature-extractor)
├─ docker/        # Docker Compose
├─ datasets/      # Datasets de entrenamiento (sprints futuros)
├─ models/        # Modelos de IA (sprints futuros)
└─ docs/          # PRD, reglas de desarrollo y SDD
```

## Inicio rápido

1. Copiar la configuración de entorno:

   ```
   cp .env.example .env
   ```

2. Instalar dependencias y arrancar el frontend:

   ```
   pnpm install
   pnpm web:dev
   ```

3. Preparar y arrancar el backend:

   ```
   uv sync --project apps/backend
   uv run --project apps/backend uvicorn app.main:app --reload --port 8000
   ```

Comandos disponibles desde la raíz:

| Comando | Acción |
|---------|--------|
| `pnpm web:dev` | Dev server de la web (puerto 5173) |
| `pnpm web:build` | Build de producción de la web |
| `pnpm backend:test` | Test del backend (pytest) |
| `pnpm backend:lint` | Lint del backend (ruff) |
| `pnpm backend:typecheck` | Typecheck del backend (mypy) |
| `pnpm desktop:dev` | App de escritorio (requiere Rust) |

## Backend con Docker

La base de datos llegará en un sprint posterior. Para levantar únicamente el backend:

```
docker compose -f docker/compose.yaml up --build
```

El backend responde en `http://localhost:8000/api/healthz`.

## Configuración por variables de entorno

Toda la configuración se hace mediante variables de entorno. Ver `.env.example` para la lista completa con sus valores por defecto.

## Modelos (Vision Engine)

Los modelos de visión se descargan con un script reproducible y se almacenan en `models/` (ignorado por git; el manifest con los hashes vive en `scripts/` y sí se commitea).

Descargar (o verificar) todos los modelos:

```
uv run --no-project --python 3.12 scripts/fetch_models.py
```

El script solo descarga lo que falta o está corrupto (valida SHA-256 contra el manifest); verificación sin descarga para CI:

```
uv run --no-project --python 3.12 scripts/fetch_models.py --check
```

Modelos en uso actualmente:

| Modelo | Archivo | Destino |
|--------|---------|---------|
| face_detector | `face_detector.tflite` (BlazeFace short range) | `models/vision/face_detector.tflite` |
| face_landmarker | `face_landmarker.task` (468 puntos de referencia) | `models/vision/face_landmarker.task` |

Para agregar un modelo nuevo: añadir su entrada (nombre, subdirectorio, archivo, URL y SHA-256) a `scripts/models_manifest.json` y volver a correr el script. Los adaptadores del Vision Engine reciben la ruta del modelo por constructor, así que la gestión de modelos es independiente del código de visión.

**Modo offline:** una vez descargados, no se necesita red. El script salta los modelos ya presentes con hash válido, y `--check` solo verifica lo que hay en disco.

## Sprint 1 — ya implementado

- **Vision Engine** (`packages/vision-engine`): cámara (OpenCV, DSHOW-first en Windows), detección facial y malla de 468 puntos (MediaPipe), pipeline por estados.
- **Feature Extractor** (`packages/feature-extractor`): geometría facial (EAR, MAR, iris, head pose) y ventana deslizante de 30 s con agregados temporales.
- **AI Engine** (`packages/ai-engine`): heurística de riesgo (S1-17) con escalera de estados `normal → long_blink → fatigue_warning → microsleep_suspected → critical_microsleep` y puntaje 0–100; seam para un modelo entrenado futuro.
- **Backend** (`apps/backend`): FastAPI + WebSocket con overlay por frame, assessments ~10 Hz, eventos biométricos, ping/keep-alive y reconexión.
- **Frontend** (`apps/web`): dashboard en tiempo real con gauge, métricas, RiskBanner, overlay facial (parpadeo, bostezo, dirección de iris, head pose), espejo configurable, alertas temporales y sonido por severidad (S1-19).
- **Contratos compartidos** (`packages/contracts`): esquema de mensajes WS y eventos sincronizado entre TS y Python.

Pendiente para el Sprint 2: base de datos PostgreSQL, modelo de IA entrenado, packaging desktop y entrenamiento con datasets reales.