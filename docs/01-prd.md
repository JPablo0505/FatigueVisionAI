# Product Requirements Document (PRD)

# FatigueVision AI

**Version:** 1.0

---

# 1. Overview

FatigueVision AI es una plataforma inteligente de monitoreo de fatiga mediante visión artificial.

El sistema utilizará una cámara convencional para detectar signos de fatiga y microsueños en tiempo real mediante técnicas de Computer Vision y Machine Learning ejecutadas localmente.

El proyecto deberá funcionar sin conexión a Internet y no dependerá de APIs externas de Inteligencia Artificial para su funcionamiento.

La Inteligencia Artificial generativa únicamente será utilizada durante el desarrollo del software para acelerar la arquitectura, documentación e implementación.

---

# 2. Objetivo

Construir una plataforma moderna, escalable y multiplataforma capaz de detectar niveles de fatiga utilizando visión artificial en tiempo real.

El proyecto deberá poder ejecutarse tanto como aplicación web como aplicación de escritorio compartiendo la mayor cantidad posible de código.

---

# 3. Problema

La fatiga y los microsueños representan un riesgo importante para conductores, estudiantes, operadores industriales y cualquier persona que permanezca largos periodos realizando actividades que requieren atención constante.

Las soluciones existentes suelen requerir hardware especializado, ser costosas o depender de servicios en la nube.

FatigueVision AI busca ofrecer una alternativa accesible, ejecutándose completamente de forma local utilizando únicamente una cámara convencional.

---

# 4. Objetivos del MVP

El MVP deberá permitir:

* Capturar video desde una cámara web.
* Detectar el rostro del usuario.
* Obtener landmarks faciales mediante MediaPipe Face Mesh.
* Calcular indicadores de fatiga (en sprints posteriores).
* Ejecutar un modelo de Machine Learning local.
* Mostrar un dashboard en tiempo real.
* Generar alertas cuando exista riesgo de fatiga.
* Registrar eventos detectados.

---

# 5. Alcance

## Incluye

* Aplicación Web.
* Aplicación de Escritorio.
* Backend API.
* Motor de Visión Artificial.
* Motor de IA.
* Dashboard.
* Base de datos.
* Docker.

## No incluye

* Aplicación móvil.
* Autenticación.
* Usuarios múltiples.
* Servicios Cloud.
* Dependencia de APIs de IA.
* Integración con vehículos.

---

# 6. Usuarios Objetivo

* Conductores.
* Estudiantes.
* Operadores de maquinaria.
* Investigadores.
* Desarrolladores interesados en visión artificial.

---

# 7. Tecnologías

## Frontend

* React
* TypeScript
* Vite
* Tailwind CSS

## Desktop

* Tauri

## Backend

* FastAPI
* Python

## Visión Artificial

* OpenCV
* MediaPipe

## Machine Learning

* Scikit-learn
* ONNX Runtime

## Base de Datos

* PostgreSQL

## Infraestructura

* Docker
* Docker Compose

---

# 8. Datasets

El entrenamiento del modelo utilizará múltiples datasets públicos para mejorar la capacidad de generalización.

Datasets previstos:

* NTHU Driver Drowsiness Detection Dataset
* YawDD
* MRL Eye Dataset
* Closed Eyes In The Wild (CEW)
* UTA-RLDD

Todos los datasets deberán normalizarse antes del entrenamiento.

---

# 9. Principios del Proyecto

* Arquitectura modular.
* Procesamiento completamente local.
* Sin dependencia de servicios externos.
* Código reutilizable.
* Escalable.
* Multiplataforma.
* Preparado para futuras mejoras.

---

# 10. Roadmap

## Sprint 1

Infraestructura base.

* Monorepo.
* Frontend.
* Backend.
* Docker.
* Tauri.
* Cámara.
* Face Detection.
* Face Mesh.

## Sprint 2

Motor de visión.

* EAR.
* MAR.
* Head Pose.
* Blink Rate.
* PERCLOS.
* Dashboard.

## Sprint 3

Machine Learning.

* Pipeline de datasets.
* Entrenamiento.
* Exportación a ONNX.
* Integración del modelo.

## Sprint 4

Sistema de fatiga.

* Índice de riesgo.
* Alertas.
* Historial.
* Reportes.

## Sprint 5

Despliegue.

* Aplicación de escritorio.
* Publicación web.
* Optimización.
* Documentación final.

---

# 11. Criterios de Éxito

El proyecto será considerado exitoso cuando:

* Detecte correctamente el rostro en tiempo real.
* Ejecute el modelo de IA localmente.
* Muestre un nivel de fatiga comprensible para el usuario.
* Funcione tanto como aplicación web como de escritorio.
* No dependa de servicios externos para operar.
* Mantenga una arquitectura modular que facilite futuras ampliaciones.
