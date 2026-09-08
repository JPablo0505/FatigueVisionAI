# datasets

Estructura para incorporar datasets de entrenamiento de fatiga.

## Estructura

```
datasets/
├── raw/          # datos originales, sin modificar
├── processed/    # datos procesados (features extraídas)
├── metadata/     # YAML descriptors de cada dataset
└── README.md
```

## Cómo agregar un dataset

Cada dataset necesita:

1. **Datos crudos** en `datasets/raw/{nombre}/`
2. **Descriptor YAML** en `datasets/metadata/{nombre}.yaml`
3. **Un adapter** que implemente `DatasetPort` (si el formato no es estándar)

### Descriptor YAML mínimo

```yaml
name: mi-dataset
version: "1.0"
source_path: datasets/raw/mi-dataset
description: "Descripción del dataset"
license: "Licencia"
subjects: 10
total_frames: 50000
annotation_schema:
  fatigue_level:
    type: categorical
    values: [alert, drowsy, sleepy]
label_vocabulary:
  fatigue_level: [alert, drowsy, sleepy]
```

### Campos requeridos

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `name` | string | Identificador único del dataset |
| `version` | string | Versión del dataset |
| `source_path` | path | Ruta a los datos crudos (absoluta o relativa) |

### Campos opcionales

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `description` | string | Descripción del dataset |
| `license` | string | Licencia de uso |
| `citation` | string | Referencia académica |
| `subjects` | int | Número de sujetos |
| `total_frames` | int | Total de frames |
| `annotation_schema` | dict | Esquema de anotaciones disponibles |
| `label_vocabulary` | dict | Valores permitidos por etiqueta |

## Formato de datos esperado

El pipeline espera datos que puedan convertirse a `VisionEvent`:

```python
@dataclass(frozen=True, slots=True)
class VisionEvent:
    frame_id: int          # ID correlativo del frame
    timestamp_ns: int      # Epoch en nanosegundos
    source: str            # Tipo de fuente ("opencv", "synthetic")
    detections: list[FaceDetection]  # Detecciones de caras
    meshes: list[FaceLandmarks]      # Mallas de landmarks (478 puntos)
```

Cada dataset debe proveer la información necesaria para construir estos objetos.

## Uso del Dataset Manager

```python
from dataset_manager import DatasetManager
from dataset_manager.adapters import SyntheticDatasetAdapter

# Listar datasets registrados
manager = DatasetManager()
print(manager.list_datasets())

# Procesar un dataset
adapter = SyntheticDatasetAdapter()
for sample in manager.process(adapter):
    # sample.features: FatigueFeatures | None
    # sample.labels: dict con ground-truth
    print(sample.labels["fatigue_level"])

# Guardar a archivo
manager.process_to_file(adapter, Path("processed/synthetic.jsonl"))
```

## Crear un adapter para un dataset nuevo

Implementa `DatasetPort`:

```python
from dataset_manager.types import DatasetPort, DatasetInfo
from vision_engine.types import VisionEvent

class MiDatasetAdapter(DatasetPort):
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="mi-dataset",
            version="1.0",
            source_path=Path("/ruta/a/datos"),
        )

    def validate(self) -> list[str]:
        # Validar que los datos estén bien formados
        errors = []
        # ... validación ...
        return errors

    def iter_events(self):
        # Iterar sobre samples y yield (VisionEvent, labels)
        for frame_data in self._load_frames():
            event = self._to_vision_event(frame_data)
            labels = {"fatigue_level": frame_data["label"]}
            yield event, labels
```

## Notas

- Los datos en `raw/` son inmutables — nunca modificarlos directamente.
- Los datos procesados se guardan en `processed/` con formato JSONL.
- El `HeuristicFatigueModel` NO se reemplaza — esto es solo para entrenamiento futuro.
- No se descargan datasets externos automáticamente — todo es manual y reproducible.

---

## YAWDD (Yawning Detection Dataset)

### Fuente

Kaggle: https://www.kaggle.com/datasets/dmfabyth/yawdd
IEEE DataPort: https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset

### Estructura del dataset

```
datasets/raw/yawdd/
├── Mirror/                    # Cámara bajo el espejo retrovisor (~322 videos)
│   ├── 1-FemaleNoGlasses-Talking.avi
│   ├── 1-FemaleNoGlasses-Yawning.avi
│   ├── 1-FemaleNoGlasses-Normal.avi
│   └── ...
└── Dash/                      # Cámara en el dashboard (~29 videos, 1 por sujeto)
    ├── 1-FemaleNoGlasses-Talking.avi
    └── ...
```

### Convención de nombres

Los nombres de archivo codifican la condición:

| Patrón | Ejemplo | Significado |
|--------|---------|-------------|
| `{id}-{gender}{glasses}-{condition}.avi` | `1-FemaleNoGlasses-Talking.avi` | Sujeto 1, mujer sin lentes, hablando |
| `{id}-{gender}{glasses}-Yawning.avi` | `5-MaleGlasses-Yawning.avi` | Sujeto 5, hombre con lentes, bostezando |
| `{id}-{gender}{glasses}-Normal.avi` | `10-FemaleNoGlasses-Normal.avi` | Sujeto 10, mujer sin lentes, silenciosa |

Condiciones posibles en el nombre: `Talking`, `Yawning`, `Normal`, `Singing`.

### Formato de video

- Formato: AVI (MJPG codec)
- Resolución: 640×480 px
- FPS: 30
- Audio: sin audio

### Uso

```python
from dataset_manager.adapters import YAWDDAdapter, YAWDDConfig
from dataset_manager import DatasetManager

config = YAWDDConfig(
    yawdd_root=Path("datasets/raw/yawdd"),
    sets="mirror",      # "mirror", "dash", o "both"
    frame_skip=1,       # Procesar cada N-ésimo frame
)
adapter = YAWDDAdapter(config)

# Validar
errors = adapter.validate()
if errors:
    for e in errors:
        print(f"  ERROR: {e}")

# Procesar
manager = DatasetManager()
for sample in manager.process(adapter):
    # sample.features: FatigueFeatures | None
    # sample.labels: dict con subject_id, gender, glasses, condition, etc.
    print(sample.labels["condition"])
```

### Limitaciones

1. **No hay anotaciones por frame**: YAWDD no tiene labels de fatiga a nivel de frame. Las condiciones (normal/talking/yawning) se infieren del nombre del archivo y se asignan a todos los frames de ese video.

2. **No hay label `fatigue_level`**: El dataset solo distingue entre normal, talking y yawning. Los indicadores fisiológicos (EAR, MAR, PERCLOS, etc.) se computan por el `FeatureExtractor` del pipeline.

3. **Caras no siempre visibles**: En algunos frames puede no haber caras detectadas. El adapter usa MediaPipe internamente para face detection y mesh estimation.

4. **Permisos académicos**: El dataset está sujeto a los términos de IEEE DataPort. No se distribuye ni descarga automáticamente.

### Archivos relacionados

- `packages/dataset-manager/dataset_manager/adapters/yawdd.py` — Adapter
- `packages/dataset-manager/tests/test_yawdd.py` — Tests (21 casos)
- `datasets/metadata/yawdd.yaml` — Descriptor YAML
