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

- Carga de `.onnx`, ONNX Runtime ni TensorFlow.

## Modelo ML de Producción (S2-08)

El paquete incluye un modelo entrenado listo para inferencia:

```python
from ai_engine import FatigueFeatures, FatiguePipeline, HeadPose, IrisDirection
from ai_engine.ml_model import MLFatigueModel

# Carga automática del modelo promovido
model = MLFatigueModel()

features = FatigueFeatures(
    ear=0.32, mar=0.18, perclos=0.41, blink_rate=18.5,
    head_pose=HeadPose(yaw=0.1, pitch=0.2, roll=0.3),
    eye_closure_duration=0.4, yawn_duration=1.2,
    iris_direction=IrisDirection(x=0.5, y=-0.25),
    window_seconds=30.0,
)

result = model.predict(features)
print(result.score)      # 0-100
print(result.level)      # low / medium / high
print(result.confidence) # 0.0-1.0
print(result.state)      # normal / fatigue_warning / microsleep_suspected
```

### Detalles del Modelo

| Campo | Valor |
|---|---|
| Versión | `fatigue_v1` |
| Algoritmo | RandomForestClassifier (100 árboles, depth=10) |
| Datos de entrenamiento | YAWDD — 117 ventanas, 32 sujetos |
| Features | 20 (EAR, MAR, PERCLOS, blink rate, yawn, head pose, iris) |
| Validación | GroupKFold CV K=5: **94.9% accuracy, 88.3% F1-macro** |
| Test set | **100% accuracy** (25 ventanas, 8 sujetos) |
| Latencia | < 20ms por inferencia (single-threaded Python) |
| Artefacto | `ai_engine/models/fatigue_v1.joblib` (160 KB) |

### Artefacto del Modelo

El archivo `fatigue_v1.joblib` contiene:

```python
{
    "model": RandomForestClassifier,       # modelo ajustado
    "feature_cols": list[str],             # 20 nombres de features
    "target_map": dict[str, str],          # fatigue_state -> FatigueLevel
    "class_labels": list[str],             # ["high", "low", "medium"]
    "hyperparameters": dict,               # hiperparámetros de entrenamiento
    "experiment_timestamp": str,           # timestamp de la corrida
}
```

### Clases de FatigueModel

| Clase | Tipo | Uso |
|---|---|---|
| `MLFatigueModel` | ML (production) | Modelo entrenado para inferencia real |
| `BaselineFatigueModel` | ML (baseline) | Versión anterior para referencia |
| `HeuristicFatigueModel` | Heurístico | Reglas basadas en umbrales |
| `StubFatiguePredictor` | Stub | Placeholder para testing |

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
