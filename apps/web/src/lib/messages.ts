import { MessageKind } from '@fatiguevision/contracts'
import type {
  BiometricEvent,
  FaceOverlay,
  FaceOverlayFace,
  FatigueFeatures,
  RiskState,
  ServerToClientMessage,
} from '@fatiguevision/contracts'

const SERVER_KINDS: ReadonlySet<string> = new Set([
  MessageKind.Pong,
  MessageKind.VisionStatus,
  MessageKind.FaceOverlay,
  MessageKind.FatigueAssessment,
  MessageKind.BiometricEvent,
  MessageKind.Error,
])

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isFiniteNumbers(obj: Record<string, unknown>, keys: readonly string[]): boolean {
  return keys.every((key) => isFiniteNumber(obj[key]))
}

/**
 * Valida y tipa un mensaje server->client recibido por el WebSocket.
 *
 * Devuelve `null` si el payload no respeta el contrato (mensaje desconocido,
 * campos con tipo incorrecto, etc.) para que el consumidor lo descarte sin
 * romper el stream. No hay `any` deliberado: el parsing reduce desde
 * `unknown` con guards de tipo.
 */
export function parseServerMessage(raw: unknown): ServerToClientMessage | null {
  let parsed: unknown
  try {
    parsed = JSON.parse(String(raw))
  } catch {
    return null
  }
  if (!isRecord(parsed)) return null
  if (typeof parsed.messageId !== 'string') return null
  if (!isFiniteNumber(parsed.timestamp)) return null
  if (typeof parsed.protocolVersion !== 'string') return null
  const kind = parsed.kind
  if (typeof kind !== 'string' || !SERVER_KINDS.has(kind)) return null
  const payload = parsed.payload
  if (!isRecord(payload)) return null

  const envelope = {
    messageId: parsed.messageId,
    timestamp: parsed.timestamp,
    protocolVersion: parsed.protocolVersion,
  }

  switch (kind) {
    case MessageKind.Pong: {
      if (typeof payload.replyTo !== 'string') return null
      return { ...envelope, kind: MessageKind.Pong, payload: { replyTo: payload.replyTo } }
    }
    case MessageKind.VisionStatus: {
      const pipeline = payload.pipeline
      if (pipeline !== 'running' && pipeline !== 'stopped' && pipeline !== 'error') return null
      const camera = payload.camera
      if (!isRecord(camera)) return null
      const source = camera.source
      if (source !== 'opencv' && source !== 'synthetic' && source !== 'browser') return null
      if (typeof camera.running !== 'boolean') return null
      const resolution = camera.resolution
      let parsedResolution: { width: number; height: number } | undefined
      if (resolution !== undefined) {
        if (!isRecord(resolution)) return null
        if (!isFiniteNumbers(resolution, ['width', 'height'])) return null
        parsedResolution = { width: resolution.width as number, height: resolution.height as number }
      }
      const fps = camera.fps
      if (fps !== undefined && !isFiniteNumber(fps)) return null
      const error = payload.error
      if (error !== undefined && typeof error !== 'string') return null
      return {
        ...envelope,
        kind: MessageKind.VisionStatus,
        payload: {
          camera: {
            source: source,
            running: camera.running,
            ...(parsedResolution !== undefined ? { resolution: parsedResolution } : {}),
            ...(fps !== undefined ? { fps: fps } : {}),
          },
          pipeline: pipeline,
          ...(error !== undefined ? { error: error } : {}),
        },
      }
    }
    case MessageKind.FatigueAssessment: {
      if (!isFiniteNumbers(payload, ['score', 'confidence', 'timestamp'])) return null
      const level = payload.level
      if (level !== 'low' && level !== 'medium' && level !== 'high') return null
      const features = payload.features
      let parsedFeatures: FatigueFeatures | undefined
      if (features !== undefined) {
        if (!isRecord(features)) return null
        if (!isFiniteNumbers(features, ['ear', 'mar', 'perclos', 'blink_rate', 'eye_closure_duration', 'yawn_duration', 'window_seconds'])) return null
        const headPose = features.head_pose
        const irisDirection = features.iris_direction
        if (!isRecord(headPose) || !isFiniteNumbers(headPose, ['yaw', 'pitch', 'roll'])) return null
        if (!isRecord(irisDirection) || !isFiniteNumbers(irisDirection, ['x', 'y'])) return null
        // Campos ADITIVOS (S1-16): opcionales en el cable (servidores
        // anteriores no los envían). Ausentes o no numéricos => 0, para que el
        // dashboard los muestre siempre sin chequeos de undefined.
        const currentYawnDuration = isFiniteNumber(features.current_yawn_duration) ? features.current_yawn_duration : 0
        const lastYawnDuration = isFiniteNumber(features.last_yawn_duration) ? features.last_yawn_duration : 0
        const marInstant = isFiniteNumber(features.mar_instant) ? features.mar_instant : 0
        // Campos ADITIVOS (S1-17): mismo criterio de tolerancia.
        const earInstant = isFiniteNumber(features.ear_instant) ? features.ear_instant : 0
        const currentEyeClosureDuration = isFiniteNumber(features.current_eye_closure_duration)
          ? features.current_eye_closure_duration
          : 0
        const longestEyeClosureDuration = isFiniteNumber(features.longest_eye_closure_duration)
          ? features.longest_eye_closure_duration
          : 0
        const longestYawnDuration = isFiniteNumber(features.longest_yawn_duration) ? features.longest_yawn_duration : 0
        const totalBlinks = isFiniteNumber(features.total_blinks) ? features.total_blinks : 0
        const totalYawns = isFiniteNumber(features.total_yawns) ? features.total_yawns : 0
        parsedFeatures = {
          ear: features.ear as number,
          mar: features.mar as number,
          mar_instant: marInstant,
          perclos: features.perclos as number,
          blink_rate: features.blink_rate as number,
          head_pose: {
            yaw: headPose.yaw as number,
            pitch: headPose.pitch as number,
            roll: headPose.roll as number,
          },
          eye_closure_duration: features.eye_closure_duration as number,
          yawn_duration: features.yawn_duration as number,
          current_yawn_duration: currentYawnDuration,
          last_yawn_duration: lastYawnDuration,
          iris_direction: {
            x: irisDirection.x as number,
            y: irisDirection.y as number,
          },
          window_seconds: features.window_seconds as number,
          ear_instant: earInstant,
          current_eye_closure_duration: currentEyeClosureDuration,
          longest_eye_closure_duration: longestEyeClosureDuration,
          longest_yawn_duration: longestYawnDuration,
          total_blinks: totalBlinks,
          total_yawns: totalYawns,
        }
      }
      // Estado de riesgo ADITIVO (S1-17): ausente o inválido => "normal".
      const state = payload.state
      const riskState: RiskState =
        state === 'long_blink' ||
        state === 'fatigue_warning' ||
        state === 'microsleep_suspected' ||
        state === 'critical_microsleep'
          ? state
          : 'normal'
      return {
        ...envelope,
        kind: MessageKind.FatigueAssessment,
        payload: {
          score: payload.score as number,
          level: level,
          confidence: payload.confidence as number,
          timestamp: payload.timestamp as number,
          state: riskState,
          ...(parsedFeatures !== undefined ? { features: parsedFeatures } : {}),
        },
      }
    }
    case MessageKind.BiometricEvent: {
      const type = payload.type
      if (type !== 'blink' && type !== 'long_blink' && type !== 'eyes_closed' && type !== 'yawn') return null
      if (!isFiniteNumbers(payload, ['timestamp', 'duration_seconds'])) return null
      const event: BiometricEvent = {
        type: type,
        timestamp: payload.timestamp as number,
        duration_seconds: payload.duration_seconds as number,
      }
      return { ...envelope, kind: MessageKind.BiometricEvent, payload: event }
    }
    case MessageKind.FaceOverlay: {
      // frame_id / timestamp_ns son números finitos; face_count es number.
      if (!isFiniteNumbers(payload, ['frame_id', 'timestamp_ns', 'face_count'])) return null
      const faces = payload.faces
      if (!Array.isArray(faces)) return null
      const parsedFaces: FaceOverlayFace[] = []
      for (const face of faces) {
        if (!isRecord(face)) return null
        if (!isFiniteNumber(face.score)) return null
        // bbox: tupla [x, y, w, h] normalizada en [0, 1].
        const bbox = face.bbox
        if (!Array.isArray(bbox) || bbox.length !== 4) return null
        if (!bbox.every((value) => isFiniteNumber(value) && value >= 0 && value <= 1)) return null
        // points: array de tuplas [x, y, z] con números finitos.
        const points = face.points
        if (!Array.isArray(points)) return null
        const parsedPoints: Array<[number, number, number]> = []
        for (const point of points) {
          if (!Array.isArray(point) || point.length !== 3) return null
          if (!point.every((value) => isFiniteNumber(value))) return null
          parsedPoints.push([point[0] as number, point[1] as number, point[2] as number])
        }
        // Campos ADITIVOS (S1-18): opcionales en el cable (servidores
        // anteriores no los envían). Ausentes o con tipo inválido => se omiten,
        // para que el overlay siga funcionando con servidores viejos.
        const eyesClosed = face.eyes_closed
        const yawning = face.yawning
        const irisDirection = face.iris_direction
        const headPose = face.head_pose
        const parsedIrisDirection =
          isRecord(irisDirection) && isFiniteNumbers(irisDirection, ['x', 'y'])
            ? { x: irisDirection.x as number, y: irisDirection.y as number }
            : undefined
        const parsedHeadPose =
          isRecord(headPose) && isFiniteNumbers(headPose, ['yaw', 'pitch', 'roll'])
            ? {
                yaw: headPose.yaw as number,
                pitch: headPose.pitch as number,
                roll: headPose.roll as number,
              }
            : undefined
        parsedFaces.push({
          bbox: [bbox[0] as number, bbox[1] as number, bbox[2] as number, bbox[3] as number],
          score: face.score as number,
          points: parsedPoints,
          ...(eyesClosed === true ? { eyes_closed: true } : {}),
          ...(yawning === true ? { yawning: true } : {}),
          ...(parsedIrisDirection !== undefined ? { iris_direction: parsedIrisDirection } : {}),
          ...(parsedHeadPose !== undefined ? { head_pose: parsedHeadPose } : {}),
        })
      }
      const overlay: FaceOverlay = {
        frame_id: payload.frame_id as number,
        timestamp_ns: payload.timestamp_ns as number,
        face_count: payload.face_count as number,
        faces: parsedFaces,
      }
      return { ...envelope, kind: MessageKind.FaceOverlay, payload: overlay }
    }
    case MessageKind.Error: {
      if (typeof payload.code !== 'string') return null
      if (typeof payload.message !== 'string') return null
      const details = payload.details
      if (details !== undefined && !isRecord(details)) return null
      return {
        ...envelope,
        kind: MessageKind.Error,
        payload: {
          code: payload.code,
          message: payload.message,
          ...(details !== undefined ? { details: details } : {}),
        },
      }
    }
    default:
      return null
  }
}