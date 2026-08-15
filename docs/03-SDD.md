# Software Design Document (SDD)

# FatigueVision AI

Versión: 1.0

---

# 1. Introducción

## Propósito

Este documento describe la arquitectura técnica de FatigueVision AI.

Su objetivo es servir como guía para el desarrollo del proyecto durante todos los sprints.

---

# 2. Objetivos de Arquitectura

La arquitectura deberá cumplir los siguientes objetivos:

- Modularidad
- Escalabilidad
- Bajo acoplamiento
- Alta cohesión
- Reutilización de código
- Compatibilidad Web y Desktop
- Procesamiento local

---

# 3. Arquitectura General

El sistema estará compuesto por cinco módulos principales:

- Frontend
- Backend
- Vision Engine
- AI Engine
- Base de Datos

Todos los módulos deberán comunicarse mediante interfaces claramente definidas.

---

# 4. Arquitectura del Proyecto

fatiguevision-ai/

apps/

- web
- desktop
- backend

packages/

- vision-engine
- ai-engine
- shared
- ui

docs/

datasets/

models/

scripts/

docker/

---

# 5. Descripción de Módulos

## Frontend

Responsable de:

- interfaz gráfica
- dashboard
- visualización de cámara
- comunicación con la API

No contendrá lógica de visión artificial.

---

## Backend

Responsable de:

- API REST
- WebSockets
- persistencia
- coordinación del sistema

---

## Vision Engine

Responsable de:

- acceso a cámara
- OpenCV
- MediaPipe
- detección facial
- extracción de landmarks

No conocerá el modelo de IA.

---

## AI Engine

Responsable de:

- cargar el modelo ONNX
- ejecutar inferencias
- calcular el nivel de fatiga

No accederá directamente a la cámara.

---

## Base de Datos

Responsable de almacenar:

- eventos
- configuraciones
- estadísticas

---

# 6. Flujo General

Cámara

↓

Vision Engine

↓

Extracción de características

↓

AI Engine

↓

Resultado

↓

Dashboard

↓

Persistencia

---

# 7. Sprint 1

Durante este sprint únicamente se implementarán:

- Monorepo
- Frontend
- Backend
- Docker
- Tauri
- Cámara
- Face Detection
- Face Mesh

No se implementará todavía el modelo de IA.

---

# 8. Contrato entre módulos

El Vision Engine expone solo sus contratos: `CameraSource`, `FaceDetector`, `FaceMeshEstimator`, `FrameProcessor` y el evento de salida `VisionEvent` (serializable a JSON, sin objetos numpy).

Regla: el Vision Engine nunca importa ni conoce al AI Engine; el AI Engine nunca accede a la cámara; el Backend es quien orquesta el flujo (loop de captura, transporte, persistencia).

Evolución sin cambios de contrato: OpenCV y MediaPipe se integran como adaptadores de estos puertos en sprints posteriores.

---

# 9. Evolución

Las siguientes versiones incorporarán:

- EAR
- MAR
- PERCLOS
- Head Pose
- Blink Rate
- Entrenamiento del modelo
- Historial
- Alertas
- Reportes