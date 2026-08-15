/**
 * Constantes puras del protocolo de comunicación.
 *
 * Este módulo SOLO aporta constantes y tipos del protocolo. No contiene
 * código ejecutable: ningún valor se calcula en runtime.
 */

/**
 * Versión del protocolo de comunicación.
 *
 * Hacer bump cuando un cambio rompe la compatibilidad con versiones
 * anteriores del cable: renombrar/eliminar campos del envelope o de un
 * payload, cambiar el significado de un `kind`, etc. Los cambios aditivos
 * (agregar mensajes o campos opcionales) NO requieren bump.
 */
export const PROTOCOL_VERSION = "1";

/** Ruta canónica del API HTTP, usada tanto por el frontend como por el backend. */
export const API_PATH = "/api";

/** Ruta canónica del WebSocket, usada tanto por el frontend como por el backend. */
export const WS_PATH = "/ws";

/**
 * Kinds de mensajes WebSocket del protocolo.
 *
 * Cada valor es la cadena exacta que viaja en el campo `kind` del envelope.
 * Los mensajes client->server y server->client se discriminan por este valor.
 */
export const MessageKind = {
  VisionStatus: "vision.status",
  FatigueAssessment: "fatigue.assessment",
  BiometricEvent: "biometric.event",
  FaceOverlay: "face.overlay",
  Error: "error",
  Ping: "ping",
  Pong: "pong",
  ControlStart: "control.start",
  ControlStop: "control.stop",
  FrameProvide: "frame.provide",
} as const;

/**
 * Nombre de cada kind del protocolo.
 *
 * El doble uso del nombre (const + type) es intencional: la const es el
 * vocabulario en runtime y el type es la unión de literales del mismo
 * conjunto. Es un patrón habitual en contratos tipo-kv.
 */
export type MessageKind = (typeof MessageKind)[keyof typeof MessageKind];

/**
 * Nivel de fatiga producido por el AI Engine.
 *
 * Vocabulario acoplado 1:1 con el enum Python `ai_engine.FatigueLevel`
 * (low / medium / high). NO inventar valores adicionales: un cambio acá
 * debe reflejarse en el enum Python y viceversa.
 */
export type FatigueLevel = "low" | "medium" | "high";