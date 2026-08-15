# Vision Engine

Motor de visión de FatigueVision AI: acceso a cámara, detección facial y estimación de landmarks. Es una librería síncrona pura, sin lógica de negocio y desacoplada del AI Engine. Expone contratos (`CameraSource`, `FaceDetector`, `FaceMeshEstimator`, `FrameProcessor`) y un evento de salida `VisionEvent` serializable a JSON (sin objetos numpy).

Estado actual (S1-07): implementados los adaptadores MediaPipe Tasks (`MediaPipeFaceDetector` con BlazeFace y `MediaPipeFaceMesh` con Face Landmarker) sobre los contratos `FaceDetector`/`FaceMeshEstimator`. La conversión BGR→RGB es interna a cada adaptador; los landmarks se entregan normalizados (x, y, z en 0..1) y los bboxes en píxeles. Las fuentes de cámara (`OpenCVCameraSource`, `SyntheticCameraSource`) y los overlays puros de `drawing.py` (`draw_detections`, `draw_mesh`, sin GUI) completan el paquete.

Los modelos requeridos viven en `models/vision/` en la raíz del repo (`face_detector.tflite` y `face_landmarker.task`); el script de descarga formal llega en S1-08.

## Verificación

```sh
uv sync
uv run ruff check vision_engine tests
uv run mypy vision_engine
uv run pytest
```

Los tests de integración de MediaPipe se saltan si los modelos no están presentes.