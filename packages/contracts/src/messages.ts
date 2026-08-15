/**
 * Esquemas de mensajes WebSocket del protocolo.
 *
 * TODOS los mensajes llevan el MISMO encabezado (`MessageEnvelope`), con un
 * solo canon de tiempo: epoch en SEGUNDOS (float) — la misma unidad que
 * `time.time()` en Python y `Date.now()/1000` en el frontend. UN solo canon
 * en el protocolo.
 *
 * El protocolo es la unión discriminada por `kind`. Agregar un mensaje =
 * agregar un miembro en las uniones de abajo (único lugar).
 */

import type { MessageKind } from "./constants.js";
import type { BiometricEvent, FaceOverlay, FatigueAssessment } from "./events.js";
import type {
  ControlPayload,
  ErrorPayload,
  FrameProvidePayload,
  PingPayload,
  PongPayload,
  VisionStatus,
} from "./models.js";

/**
 * Encabezado común de TODOS los mensajes del protocolo (client<->server).
 *
 * `messageId` / `timestamp` / `protocolVersion` son generados por el EMISOR;
 * el `kind` discrimina qué payload lleva.
 */
export interface MessageEnvelope<K extends MessageKind, P> {
  /** ID único generado por el EMISOR (capa backend o web) para correlación. */
  messageId: string;
  /** Epoch en SEGUNDOS (float) al emitir — misma unidad que time.time() y Date.now()/1000. */
  timestamp: number;
  /** Versión del protocolo (= PROTOCOL_VERSION desde constants.ts). */
  protocolVersion: string;
  /** Kind del mensaje: discrimina el tipo de `payload`. */
  kind: K;
  /** Payload específico de cada kind. */
  payload: P;
}

/**
 * Mensajes que el cliente (web / desktop) envía al servidor.
 *
 * Ping = keep-alive/configuración; control.start / control.stop = arrancar /
 * detener el pipeline; frame.provide = frame de cámara del navegador (modo
 * ``camera.source == "browser"``). El control no lleva payload (la acción
 * está en `kind`).
 */
export type ClientToServerMessage =
  | MessageEnvelope<typeof MessageKind.Ping, PingPayload>
  | MessageEnvelope<typeof MessageKind.ControlStart, ControlPayload>
  | MessageEnvelope<typeof MessageKind.ControlStop, ControlPayload>
  | MessageEnvelope<typeof MessageKind.FrameProvide, FrameProvidePayload>;

/**
 * Mensajes que el servidor envía al cliente (web / desktop).
 *
 * VisionStatus = estado del pipeline; FaceOverlay = datos de detección facial
 * para que el dashboard dibuje el overlay sobre su video local;
 * BiometricEvent = episodio biométrico completado (cierre ocular / bostezo);
 * FatigueAssessment = evaluación de fatiga; Error = error del servidor;
 * Pong = respuesta a un Ping.
 */
export type ServerToClientMessage =
  | MessageEnvelope<typeof MessageKind.Pong, PongPayload>
  | MessageEnvelope<typeof MessageKind.VisionStatus, VisionStatus>
  | MessageEnvelope<typeof MessageKind.FaceOverlay, FaceOverlay>
  | MessageEnvelope<typeof MessageKind.BiometricEvent, BiometricEvent>
  | MessageEnvelope<typeof MessageKind.FatigueAssessment, FatigueAssessment>
  | MessageEnvelope<typeof MessageKind.Error, ErrorPayload>;