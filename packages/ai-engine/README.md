# @fatiguevision/ai-engine

Motor de IA de fatiga del proyecto: define el **contrato** entre el análisis visual y el modelo que decide si el conductor está fatigado. Recibe características de alto nivel (EAR, MAR, PERCLOS, blink rate, etc.) y devuelve un puntaje de fatiga (0–100), nivel, confianza y timestamp. **Nunca accede a la cámara** y **no conoce imágenes, landmarks ni librerías de visión**.

## Por qué está desacoplado

El AI Engine no depende de nada del Vision Engine: ambos se comunican **solo a través de este contrato de datos** (`FatigueFeatures` → `FatigueResult`), lo que permite entrenar, calibrar y reemplazar el modelo sin tocar el análisis visual.

## Uso mínimo

El paquete no tiene dependencias runtime (solo stdlib). Requiere Python >= 3.12.

```python
from ai_engine import FatigueFeatures, FatiguePipeline, HeadPose, IrisDirection, StubFatiguePredictor

features = FatigueFeatures(
    ear=0.32,
    mar=0.18,
    perclos=0.41,
    blink_rate=18.5,
    head_pose=HeadPose(yaw=0.1, pitch=0.2, roll=0.3),
    eye_closure_duration=0.4,
    yawn_duration=1.2,
    iris_direction=IrisDirection(x=0.5, y=-0.25),
    window_seconds=30.0,
)

result = FatiguePipeline(StubFatiguePredictor()).assess(features)
print(result.to_dict())  # {"score": 0.0, "level": "low", "confidence": 0.0, "timestamp": ...}
```

`StubFatiguePredictor` es un placeholder determinista que valida el contrato pero **no calcula fatiga real**.

## Cómo implementar un modelo real

1. Implementar el `Protocol` `FatigueModel` con `predict(features) -> FatigueResult`.
2. Sustituir el predictor del pipeline: `pipeline.predictor = MiModelo()` (el pipeline es agnóstico del predictor).
3. El pipeline sella el `timestamp` si el modelo no lo provee; el `level` por defecto sale de `default_fatigue_level(score)` y puede sobrescribirse en la implementación.

## Qué NO está implementado aún

- Inferencia real de fatiga ni modelos entrenados.
- Carga de `.onnx`, ONNX Runtime, scikit-learn ni TensorFlow.
- Cálculo de fatiga, calibración ni umbrales definitivos.

## Tests y calidad

```bash
uv sync
uv run ruff check ai_engine tests
uv run mypy ai_engine
uv run pytest -q
```

## Contrato

| Tipo | Responsabilidad |
|------|-----------------|
| `FatigueFeatures` | Entrada: características de alto nivel del ventana de observación. |
| `FatigueResult` | Salida: score, level, confidence, timestamp. |
| `FatigueModel` | Port: mapea `FatigueFeatures` → `FatigueResult`. |
| `FatiguePipeline` | Orquestación pura: delega y sella timestamp. |
| `AIEngineError` y subclases | Errores del dominio (`InvalidFeaturesError`, `InferenceError`). |
