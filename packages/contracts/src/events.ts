/**
 * Payloads de eventos producidos por los motores (`vision_engine` /
 * `ai_engine`).
 *
 * Estas shapes espejan EXACTAMENTE los `to_dict()` de Python — esa es la
 * fuente de verdad del cable. Se especularon a partir de la lectura de:
 *   - packages/vision-engine/vision_engine/types.py
 *   - packages/ai-engine/ai_engine/types.py
 *
 * Un cambio en los `to_dict()` de Python debe reflejarse aquí (y viceversa).
 */

import type { FatigueLevel } from "./constants.js";

/**
 * Estado de riesgo en tiempo real producido por el AI Engine.
 *
 * Vocabulario acoplado 1:1 con el enum Python `ai_engine.RiskState`
 * (normal / long_blink / fatigue_warning / microsleep_suspected /
 * critical_microsleep). NO inventar valores adicionales: un cambio acá
 * debe reflejarse en el enum Python y viceversa.
 */
export type RiskState =
  | "normal"
  | "long_blink"
  | "fatigue_warning"
  | "microsleep_suspected"
  | "critical_microsleep";

/**
 * Bounding box de una cara detectada, en píxeles de la imagen.
 *
 * Espeja `FaceDetection.to_dict()` de vision_engine.
 */
export interface FaceDetection {
  /** [x1, y1, x2, y2] en píxeles. */
  bbox: [number, number, number, number];
  /** Confianza de la detección en [0, 1]. */
  score: number;
}

/**
 * Malla de puntos de referencia faciales en coordenadas normalizadas.
 *
 * Espeja `FaceLandmarks.to_dict()` de vision_engine.
 */
export interface FaceLandmarks {
  /** Lista de puntos [x, y, z] normalizados (cada punto es un array de 3 floats). */
  points: Array<[number, number, number]>;
}

/**
 * Salida del pipeline de visión para un frame: metadata + detecciones +
 * mallas.
 *
 * Espeja `VisionEvent.to_dict()` de vision_engine, campo por campo. Nota:
 * el campo de tiempo del evento es `timestamp_ns` (nanosegundos, entero),
 * distinto del `timestamp` del envelope (segundos, float).
 */
export interface VisionEvent {
  /** ID correlativo del frame dentro de la sesión del pipeline. */
  frame_id: number;
  /** Epoch en nanosegundos (entero) al capturar el frame. */
  timestamp_ns: number;
  /** Cadena del tipo de fuente de cámara ("opencv" | "synthetic"). */
  source: string;
  /** Cantidad de caras detectadas (= detections.length). */
  face_count: number;
  /** Detecciones de caras del frame. */
  detections: FaceDetection[];
  /** Mallas de landmarks del frame (una por cara detectada). */
  meshes: FaceLandmarks[];
}

/**
 * Orientación de cabeza en radianes.
 *
 * Espeja `ai_engine.HeadPose` (y `HeadPoseModel` del backend).
 */
export interface HeadPose {
  /** Yaw en radianes. */
  yaw: number;
  /** Pitch en radianes. */
  pitch: number;
  /** Roll en radianes. */
  roll: number;
}

/**
 * Dirección de la mirada (iris) normalizada en [-1, 1].
 *
 * Espeja `ai_engine.IrisDirection` (y `IrisDirectionModel` del backend).
 */
export interface IrisDirection {
  /** Componente horizontal en [-1, 1]. */
  x: number;
  /** Componente vertical en [-1, 1]. */
  y: number;
}

/**
 * Características biométricas agregadas por el Feature Extractor.
 *
 * Espeja `ai_engine.FatigueFeatures.to_dict()` (y `FatigueFeaturesModel` del
 * backend). Viaja DENTRO de `FatigueAssessment.features` para que el
 * dashboard pueda mostrar las métricas en tiempo real sin calcular nada.
 */
export interface FatigueFeatures {
  /** Eye Aspect Ratio medio de la ventana. */
  ear: number;
  /** Mouth Aspect Ratio medio de la ventana. */
  mar: number;
  /**
   * MAR instantáneo del último frame con cara.
   *
   * ADITIVO (agregado en S1-16, sin bump de PROTOCOL_VERSION): opcional en el
   * cable porque los servidores anteriores no lo emiten; un cliente nuevo
   * debe tolerar su ausencia (default 0).
   */
  mar_instant?: number;
  /** Porcentaje de cierre ocular en [0, 1]. */
  perclos: number;
  /** Parpadeos por minuto. */
  blink_rate: number;
  /** Orientación de cabeza (radianes). */
  head_pose: HeadPose;
  /** Duración media de cierres oculares en segundos. */
  eye_closure_duration: number;
  /** Duración media de bostezos (solo bostezos COMPLETADOS) en segundos. */
  yawn_duration: number;
  /**
   * Duración en segundos del bostezo ACTIVO (aún en curso) al final de la
   * ventana; 0 si el último frame no está bostezando.
   *
   * ADITIVO (agregado en S1-16, sin bump de PROTOCOL_VERSION): opcional en el
   * cable porque los servidores anteriores no lo emiten; un cliente nuevo
   * debe tolerar su ausencia (default 0).
   */
  current_yawn_duration?: number;
  /**
   * Duración en segundos del ÚLTIMO bostezo COMPLETADO; 0 si ninguno se
   * completó en la ventana.
   *
   * ADITIVO (agregado en S1-16, sin bump de PROTOCOL_VERSION): opcional en el
   * cable porque los servidores anteriores no lo emiten; un cliente nuevo
   * debe tolerar su ausencia (default 0).
   */
  last_yawn_duration?: number;
  /**
   * EAR instantáneo del último frame con cara.
   *
   * ADITIVO (agregado en S1-17, sin bump de PROTOCOL_VERSION): opcional en el
   * cable porque los servidores anteriores no lo emiten; un cliente nuevo
   * debe tolerar su ausencia (default 0).
   */
  ear_instant?: number;
  /**
   * Duración en segundos del cierre ocular ACTIVO (aún en curso) al final de
   * la ventana; 0 si el último frame no tiene los ojos cerrados.
   *
   * ADITIVO (agregado en S1-17, sin bump de PROTOCOL_VERSION): opcional en el
   * cable porque los servidores anteriores no lo emiten; un cliente nuevo
   * debe tolerar su ausencia (default 0).
   */
  current_eye_closure_duration?: number;
  /**
   * Duración en segundos del cierre ocular más largo de la SESIÓN (no se
   * reinicia con la ventana).
   *
   * ADITIVO (agregado en S1-17, sin bump de PROTOCOL_VERSION): opcional en el
   * cable porque los servidores anteriores no lo emiten; un cliente nuevo
   * debe tolerar su ausencia (default 0).
   */
  longest_eye_closure_duration?: number;
  /**
   * Duración en segundos del bostezo más largo de la SESIÓN (no se reinicia
   * con la ventana).
   *
   * ADITIVO (agregado en S1-17, sin bump de PROTOCOL_VERSION): opcional en el
   * cable porque los servidores anteriores no lo emiten; un cliente nuevo
   * debe tolerar su ausencia (default 0).
   */
  longest_yawn_duration?: number;
  /**
   * Parpadeos totales de la SESIÓN (incluye long blinks y ojos cerrados; no
   * se reinicia con la ventana).
   *
   * ADITIVO (agregado en S1-17, sin bump de PROTOCOL_VERSION): opcional en el
   * cable porque los servidores anteriores no lo emiten; un cliente nuevo
   * debe tolerar su ausencia (default 0).
   */
  total_blinks?: number;
  /**
   * Bostezos totales de la SESIÓN (no se reinicia con la ventana).
   *
   * ADITIVO (agregado en S1-17, sin bump de PROTOCOL_VERSION): opcional en el
   * cable porque los servidores anteriores no lo emiten; un cliente nuevo
   * debe tolerar su ausencia (default 0).
   */
  total_yawns?: number;
  /** Dirección de la mirada en [-1, 1]. */
  iris_direction: IrisDirection;
  /** Largo de la ventana temporal en segundos. */
  window_seconds: number;
}

/**
 * Resultado de una evaluación de fatiga del AI Engine.
 *
 * Espeja `FatigueResult.to_dict()` de ai_engine. En el cable el `timestamp`
 * SIEMPRE está sellado por el backend: en Python `FatigueResult.timestamp`
 * puede ser None (se serializa como null hasta que el pipeline lo sella),
 * pero antes de emitirse a través del WebSocket el backend lo sella con
 * epoch en segundos (float). Por eso acá es obligatorio y no nullable.
 *
 * `features` es ADITIVO (agregado en S1-15, sin bump de PROTOCOL_VERSION):
 * el backend lo incluye siempre en los assessments emitidos; los clientes
 * que no lo lean siguen funcionando con los cuatro campos originales.
 */
export interface FatigueAssessment {
  /** Score de fatiga en [0, 100]. */
  score: number;
  /** Nivel de fatiga (vocabulario de `ai_engine.FatigueLevel`). */
  level: FatigueLevel;
  /** Confianza de la evaluación en [0, 1]. */
  confidence: number;
  /** Epoch en segundos (float) sellado por el backend al emitir la evaluación. */
  timestamp: number;
  /** Características biométricas que alimentaron la evaluación. */
  features?: FatigueFeatures;
  /**
   * Estado de riesgo en tiempo real (vocabulario de `ai_engine.RiskState`).
   *
   * ADITIVO (agregado en S1-17, sin bump de PROTOCOL_VERSION): opcional en el
   * cable porque los servidores anteriores no lo emiten; un cliente nuevo
   * debe tolerar su ausencia (default "normal").
   */
  state?: RiskState;
}

/**
 * Evento biométrico puntual completado por el Feature Extractor
 * (streaming, server->client).
 *
 * ADITIVO (agregado en S1-17, sin bump de PROTOCOL_VERSION): el backend lo
 * emite cuando se completa un episodio (cierre ocular o bostezo) para que el
 * dashboard muestre los eventos recientes en tiempo real. Cierres muy largos
 * (`eyes_closed`) son señal temprana de microsueño.
 */
export interface BiometricEvent {
  /** Tipo de evento: "blink" | "long_blink" | "eyes_closed" | "yawn". */
  type: string;
  /** Epoch en segundos (float) del ÚLTIMO frame del episodio. */
  timestamp: number;
  /** Duración total del episodio en segundos. */
  duration_seconds: number;
}

/**
 * Overlay de detección facial para un frame (streaming, server->client).
 *
 * ADITIVO (agregado en S1-15, sin bump de PROTOCOL_VERSION): el backend lo
 * emite a ~20 Hz mientras el pipeline corre para que el dashboard dibuje el
 * bounding box + face mesh sobre su video local (`getUserMedia`) sin que el
 * frontend calcule NADA. El frontend SOLO dibuja los datos recibidos; toda
 * la detección y las métricas viven en el backend.
 *
 * Todos los puntos vienen NORMALIZADOS en [0, 1] (coordenadas relativas al
 * frame) para que el canvas escale sobre cualquier resolución de video.
 */
export interface FaceOverlay {
  /** ID correlativo del frame dentro de la sesión del pipeline. */
  frame_id: number;
  /** Epoch en nanosegundos (entero) al capturar el frame. */
  timestamp_ns: number;
  /** Cantidad de caras detectadas (= faces.length). */
  face_count: number;
  /** Una entrada por cara detectada (vacío si no hay rostro). */
  faces: FaceOverlayFace[];
}

/**
 * Una cara dentro de un `FaceOverlay`.
 *
 * `bbox` se deriva del propio mesh (min/max de los puntos) y NO de la
 * detección corta de MediaPipe, para mantener una única fuente normalizada.
 */
export interface FaceOverlayFace {
  /** [x, y, w, h] normalizado en [0, 1] (x, y = origen superior izquierdo). */
  bbox: [number, number, number, number];
  /** Confianza de la detección en [0, 1]. */
  score: number;
  /**
   * Mesh completo de MediaPipe normalizado en [0, 1]: cada punto es
   * [x, y, z]. Índices de referencia: 4 = punta de nariz, 468-472 = iris
   * izquierdo, 473-477 = iris derecho.
   */
  points: Array<[number, number, number]>;
  /**
   * Ojos cerrados en este frame (EAR por debajo del umbral del backend).
   *
   * ADITIVO (agregado en S1-18, sin bump de PROTOCOL_VERSION): opcional en el
   * cable porque los servidores anteriores no lo emiten; el frontend solo
   * dibuja cuando viene presente.
   */
  eyes_closed?: boolean;
  /**
   * Bostezo en este frame (MAR por encima del umbral del backend).
   *
   * ADITIVO (agregado en S1-18, sin bump de PROTOCOL_VERSION): opcional en el
   * cable porque los servidores anteriores no lo emiten.
   */
  yawning?: boolean;
  /**
   * Dirección de la mirada (iris) normalizada en [-1, 1].
   *
   * ADITIVO (agregado en S1-18, sin bump de PROTOCOL_VERSION): opcional en el
   * cable porque los servidores anteriores no lo emiten.
   */
  iris_direction?: IrisDirection;
  /**
   * Orientación de cabeza (radianes).
   *
   * ADITIVO (agregado en S1-18, sin bump de PROTOCOL_VERSION): opcional en el
   * cable porque los servidores anteriores no lo emiten.
   */
  head_pose?: HeadPose;
}