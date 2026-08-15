# @fatiguevision/contracts

Contratos de comunicación compartidos entre el backend (Python, FastAPI + motores) y el frontend (TypeScript, web/desktop). El JSON que cruza el cable = los `to_dict()` de `vision_engine` / `ai_engine`; este paquete es la especificación canónica de ese cable del lado TypeScript.

**Único propósito:** tipar el protocolo. Sin funciones, sin utilidades, sin lógica de negocio — solo `type`, `interface` y `const` literales (type-only, sin dependencias runtime).

## Rápido

- ¿Necesitás contratos? Todo lo público sale de `src/index.ts`.
- ¿Agregar un mensaje nuevo? Ver [Cómo agregar un mensaje](#cómo-agregar-un-mensaje).
- ¿Cambio breaking? Ver [Evolución del protocolo](#evolución-del-protocolo).

---

## Regla de pertenencia

Algo vive en este paquete SOLO si responde SÍ a las tres preguntas:

1. **¿Cruza el cable?** ¿viaja por HTTP/WebSocket entre backend y frontend?
2. **¿Es vocabulario del protocolo?** ¿tiene nombre y semántica estables del contrato (kind, campo, constante)?
3. **¿Existiría sin el otro lado?** si lo sacás del protocolo, ¿pierde sentido?

Si alguna es NO, vive en otro lado (en el backend Python, en `apps/web`, en `apps/desktop`, etc.).

Quedan EXCLUIDOS por diseño: cualquier `function`/arrow/utility, clases, `enum`, imports de React, código ejecutable (`Date.now()` en runtime) y dependencias.

## Estructura del paquete

```
packages/contracts/
├─ package.json        # @fatiguevision/contracts, type-only, sin deps runtime
├─ tsconfig.json       # strict + bundler, extiende tsconfig.base.json
├─ README.md
└─ src/
   ├─ index.ts         # API pública (solo re-exporta tipos y constantes)
   ├─ constants.ts     # PROTOCOL_VERSION, API_PATH, WS_PATH, MessageKind, FatigueLevel
   ├─ models.ts        # shapes compartidas no-event (CameraStatus, VisionStatus, Ping/Pong...)
   ├─ events.ts        # payloads de eventos: espejo de los to_dict() de los motores
   └─ messages.ts      # MessageEnvelope + uniones client->server / server->client
```

## El envelope (encabezado común)

Todo mensaje del WebSocket usa el MISMO encabezado. **Un solo canon de tiempo** en el protocolo: epoch en **SEGUNDOS** (float) — la misma unidad que `time.time()` en Python y `Date.now()/1000` en JS.

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `messageId` | `string` | ID único generado por el **emisor** (backend o web) para correlación |
| `timestamp` | `number` | Epoch en segundos (float) al emitir |
| `protocolVersion` | `string` | `= PROTOCOL_VERSION` |
| `kind` | `MessageKind` | Discrimina el tipo de `payload` |
| `payload` | `P` | Dato específico de cada kind |

El protocolo es la **unión discriminada por `kind`**. Happy path: el cliente manda `ping` / `control.start` / `control.stop`; el servidor responde `pong`, `vision.status`, `fatigue.assessment` y `error`.

## MessageKind

| Kind | Valor | Dirección |
|------|-------|-----------|
| `MessageKind.Ping` | `"ping"` | client → server |
| `MessageKind.ControlStart` | `"control.start"` | client → server |
| `MessageKind.ControlStop` | `"control.stop"` | client → server |
| `MessageKind.Pong` | `"pong"` | server → client |
| `MessageKind.VisionStatus` | `"vision.status"` | server → client |
| `MessageKind.FatigueAssessment` | `"fatigue.assessment"` | server → client |
| `MessageKind.Error` | `"error"` | server → client |

## Mapeo al contrato Python (especificación canónica del cable)

Esta tabla es LA fuente de verdad: qué shape TS → qué `to_dict()` de Python.

| TS (este paquete) | Python (fuente de verdad) | Notas |
|-------------------|---------------------------|-------|
| `VisionEvent` | `vision_engine.types.VisionEvent.to_dict()` | Incluye `face_count`, `detections[]` y `meshes[]` |
| `FaceDetection` | `vision_engine.types.FaceDetection.to_dict()` | `bbox: [x1, y1, x2, y2]`, `score` en [0, 1] |
| `FaceLandmarks` | `vision_engine.types.FaceLandmarks.to_dict()` | `points: [[x, y, z], ...]` normalizados |
| `FatigueAssessment` | `ai_engine.types.FatigueResult.to_dict()` | `timestamp` SIEMPRE sellado en el cable (en Python puede ser `null` interno) |
| `FatigueLevel` | `ai_engine.FatigueLevel` | `"low" \| "medium" \| "high"` — sin valores inventados |
| `CameraSource` | `vision_engine.camera.factory.create_camera_source` | `"opencv" \| "synthetic"` |

Un cambio en un `to_dict()` de Python debe reflejarse acá (y viceversa).

## Evolución del protocolo

`PROTOCOL_VERSION` comienza en `"1"`.

- **Crecimiento aditivo** (nuevo mensaje, campo opcional nuevo en un payload): NO requiere bump. Se puede agregar sin romper.
- **Cambio breaking** (renombrar/eliminar un campo, cambiar el significado de un `kind`, alterar un payload existente): hacer `bump` de `PROTOCOL_VERSION`, tocar acá **y** el espejo Python en simultáneo.

Regla: el bump ES una transición coordinada. Un lado sin el otro rompe el contrato.

## Cómo agregar un mensaje

1. `src/constants.ts`: agregar el valor al const `MessageKind` (y revisar `FatigueLevel` si aplica un vocabulario nuevo).
2. `src/models.ts` o `src/events.ts`: definir el payload del nuevo mensaje (si no existe ya).
3. `src/messages.ts`: agregar el miembro en `ClientToServerMessage` o `ServerToClientMessage` — **único lugar** donde se registra el mensaje.
4. `src/index.ts`: re-exportar el tipo si es nuevo.
5. Acá (README): actualizar la tabla de mapeo al contrato Python si el payload viene de un motor.