/**
 * Modelos de datos compartidos que cruzan el cable entre el backend (Python)
 * y el frontend (TypeScript). Son las shapes canónicas del protocolo fuera
 * de los eventos de motores (ver `events.ts`).
 */

/**
 * Tipo de cámara activa.
 *
 * Vocabulario de `vision_engine.camera.factory.create_camera_source`:
 * "opencv" (dispositivo físico/virtual), "synthetic" (cámara determinista) o
 * "browser" (los frames los envía el navegador por ``frame.provide``; el
 * backend no abre ningún dispositivo).
 */
export type CameraSource = "opencv" | "synthetic" | "browser";

/** Resolución de una imagen de cámara, en píxeles. */
export interface Resolution {
  /** Ancho en píxeles. */
  width: number;
  /** Alto en píxeles. */
  height: number;
}

/** Estado reportado de una cámara. */
export interface CameraStatus {
  /** Tipo de cámara activa. */
  source: CameraSource;
  /** True si la cámara está capturando frames en este momento. */
  running: boolean;
  /** Resolución actual de la captura, si está disponible. */
  resolution?: Resolution;
  /** FPS actuales de la captura, si están disponibles. */
  fps?: number;
}

/** Estado del pipeline de visión, reportado al frontend. */
export interface VisionStatus {
  /** Estado de la cámara. */
  camera: CameraStatus;
  /** Estado general del pipeline de visión. */
  pipeline: "running" | "stopped" | "error";
  /** Mensaje de error legible, presente cuando `pipeline` es "error". */
  error?: string;
}

/** Payload de un mensaje de error: código estable + descripción legible. */
export interface ErrorPayload {
  /** Código estable de error, para lógica del cliente (no copy para UI). */
  code: string;
  /** Descripción legible del error para el usuario. */
  message: string;
  /** Detalles adicionales opcionales (estructura libre). */
  details?: Record<string, unknown>;
}

/** Payload de un ping de configuración (client->server). */
export interface PingPayload {
  /** Epoch en segundos (float) al momento de emitir el ping. */
  sentAt: number;
}

/** Payload de un pong, en respuesta a un ping. */
export interface PongPayload {
  /** `messageId` del ping que se está respondiendo. */
  replyTo: string;
}

/**
 * Payload de un control genérico.
 *
 * La acción se discrimina por el `kind` del envelope (control.start para
 * iniciar, control.stop para detener); el payload queda intencionalmente
 * vacío.
 */
export interface ControlPayload {}

/**
 * Payload de un frame de cámara enviado por el navegador (client->server).
 *
 * En modo ``camera.source == "browser"`` el backend NO abre ningún dispositivo:
 * el navegador captura con ``getUserMedia``, codifica cada frame como JPEG y
 * lo sube aquí. El backend decodifica, procesa y devuelve el overlay/assessment
 * por el mismo canal. ``width``/``height`` son los de la imagen JPEG.
 */
export interface FrameProvidePayload {
  /** ID de frame generado por el EMISOR (cliente) para correlación. */
  frameId: number;
  /** Epoch en segundos (float) al capturar el frame. */
  timestamp: number;
  /** Ancho en píxeles de la imagen JPEG. */
  width: number;
  /** Alto en píxeles de la imagen JPEG. */
  height: number;
  /** Imagen JPEG en Base64 (sin prefijo ``data:``, solo el payload). */
  jpegBase64: string;
}