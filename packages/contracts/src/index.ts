/**
 * API pública de `@fatiguevision/contracts`.
 *
 * CERO código ejecutable: solo tipos, interfaces y constantes del protocolo.
 * Sin funciones, sin utilidades, sin dependencias runtime ni de React.
 * El paquete es un lugar de pertenencia para vocabulario del cable, no para lógica.
 */

export { PROTOCOL_VERSION, API_PATH, WS_PATH, MessageKind } from "./constants.js";
export type { FatigueLevel } from "./constants.js";

export type {
  CameraSource,
  Resolution,
  CameraStatus,
  VisionStatus,
  ErrorPayload,
  PingPayload,
  PongPayload,
  ControlPayload,
  FrameProvidePayload,
} from "./models.js";

export type {
  FaceDetection,
  FaceLandmarks,
  VisionEvent,
  FaceOverlay,
  FaceOverlayFace,
  FatigueAssessment,
  FatigueFeatures,
  RiskState,
  BiometricEvent,
  HeadPose,
  IrisDirection,
} from "./events.js";

export type {
  MessageEnvelope,
  ClientToServerMessage,
  ServerToClientMessage,
} from "./messages.js";