import { useEffect, useRef } from 'react'
import type { FaceOverlay, FaceOverlayFace, HeadPose } from '@fatiguevision/contracts'
import { useBrowserFrameSender } from '../hooks/useBrowserFrameSender'
import type { FrameProvideSink } from '../hooks/useBrowserFrameSender'
import { useLanguage } from '../i18n'
import type { TranslationKey } from '../i18n'

/**
 * Toggles del overlay (S1-18). Son estado LOCAL del frontend y SOLO visuales:
 * nunca se envían al backend ni cambian qué se detecta ni qué se emite.
 */
export interface OverlayOptions {
  /** Caja delimitadora con esquinas redondeadas (brackets). */
  bbox: boolean;
  /** Puntos del mesh (468 landmarks). */
  landmarks: boolean;
  /** Resaltar ojos cerrados (usa `eyes_closed` del backend). */
  eyes: boolean;
  /** Resaltar bostezo (usa `yawning` del backend). */
  mouth: boolean;
  /** Flechas de dirección de la mirada (usa `iris_direction` del backend). */
  gaze: boolean;
  /** Gizmo de orientación de cabeza (usa `head_pose` del backend). */
  headPose: boolean;
  /** Anillo de estado alrededor de la cara (usa `assessment.state`). */
  statusRing: boolean;
  /** HUD de FPS / latencia / estado del WebSocket. */
  hud: boolean;
}

export type OverlayConnection = 'connected' | 'connecting' | 'disconnected'

/** Estado de riesgo que llega desde `assessment.state` (mapeado en App.tsx). */
interface StatusMeta {
  code: string;
  label: string;
}

interface HudData {
  fps: number;
  latencyMs: number;
  connection: OverlayConnection;
}

interface WebcamFeedProps {
  stream: MediaStream;
  overlay: FaceOverlay | null;
  /** FPS reales de llegada del overlay (medidos con useOverlayFps). */
  fps: number;
  /** Latencia del pipeline en ms (ahora - overlay.timestamp_ns). */
  latencyMs: number;
  /** Estado de la conexión WebSocket para el HUD. */
  connection: OverlayConnection;
  /** Toggles de capas del overlay (solo presentación). */
  options: OverlayOptions;
  /** Estado de riesgo (assessment.state) para el anillo de estado. */
  state?: StatusMeta | null;
  /**
   * Vista en espejo (selfie). SOLO presentación (S1-18 UX): invierte el eje X
   * de la flecha de mirada y el yaw del gizmo para que sigan la dirección
   * percibida por el usuario. No altera el backend ni el contrato.
   */
  isMirrored?: boolean;
  /** Activa el envío de frames al backend (modo camera.source == "browser"). */
  frameProvide?: FrameProvideSink | null;
}

const NOSE_INDEX = 4
const LEFT_EYE = [33, 160, 158, 133, 153, 144]
const RIGHT_EYE = [362, 385, 387, 263, 373, 380]
const LEFT_IRIS = [468, 469, 470, 471, 472]
const RIGHT_IRIS = [473, 474, 475, 476, 477]
const MOUTH_REGION = [13, 14, 61, 291]

const BBOX_COLOR = 'rgba(34, 211, 238, 0.9)'
const MESH_COLOR = 'rgba(34, 211, 238, 0.35)'
const NOSE_COLOR = 'rgba(248, 113, 113, 0.95)'
const GAZE_COLOR = 'rgba(251, 191, 36, 0.85)'
const EYE_CLOSED_FILL = 'rgba(245, 158, 11, 0.35)'
const EYE_CLOSED_STROKE = 'rgba(245, 158, 11, 0.8)'
const MOUTH_FILL = 'rgba(244, 63, 94, 0.4)'
const MOUTH_STROKE = 'rgba(244, 63, 94, 0.85)'
const HEAD_POSE_COLOR = 'rgba(96, 165, 250, 0.9)'
const HUD_BG_COLOR = 'rgba(2, 6, 23, 0.55)'
const HUD_TEXT_COLOR = 'rgba(203, 213, 225, 0.95)'
const GAZE_ARROW_PX = 40

/** Estilo visual del anillo de estado por código de riesgo (S1-18). */
const STATUS_STYLE: Record<string, { color: string; emoji: string }> = {
  normal: { color: '#10b981', emoji: '🟢' },
  long_blink: { color: '#f59e0b', emoji: '🟡' },
  fatigue_warning: { color: '#f97316', emoji: '🟠' },
  microsleep_suspected: { color: '#ef4444', emoji: '🔴' },
  critical_microsleep: { color: '#b91c1c', emoji: '🚨' },
}
const DEFAULT_STATUS_STYLE = { color: '#94a3b8', emoji: '' }

interface Size {
  width: number;
  height: number;
  dpr: number;
}

interface Region {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
}

type T = (key: TranslationKey) => string

function clamp(value: number, low: number, high: number): number {
  if (value < low) return low
  if (value > high) return high
  return value
}

function fmtDeg(value: number): string {
  return value.toFixed(1)
}

/** Promedio de un ring de índices del mesh: devuelve el punto [x, y] normalizado. */
function ringCentroid(points: Array<[number, number, number]>, indices: number[]): [number, number] | null {
  let sx = 0
  let sy = 0
  let count = 0
  for (const index of indices) {
    const point = points[index]
    if (point === undefined) continue
    const [px, py] = point
    sx += px
    sy += py
    count += 1
  }
  return count === 0 ? null : [sx / count, sy / count]
}

/** Bounding box normalizado de un conjunto de índices del mesh (para ojos/boca). */
function regionOf(points: Array<[number, number, number]>, indices: number[]): Region | null {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  let count = 0
  for (const index of indices) {
    const point = points[index]
    if (point === undefined) continue
    const [px, py] = point
    if (px < minX) minX = px
    if (px > maxX) maxX = px
    if (py < minY) minY = py
    if (py > maxY) maxY = py
    count += 1
  }
  if (count === 0) return null
  return {
    cx: (minX + maxX) / 2,
    cy: (minY + maxY) / 2,
    rx: (maxX - minX) / 2,
    ry: (maxY - minY) / 2,
  }
}

/**
 * Caja delimitadora elegante: cuatro brackets en L en las esquinas con un
 * glow sutil (shadowBlur). Reemplaza al simple strokeRect del S1-15.
 */
function drawCornerBrackets(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  color: string,
): void {
  const corner = Math.max(10, Math.min(w, h) * 0.14)
  const len = corner * 1.4
  ctx.save()
  ctx.strokeStyle = color
  ctx.lineWidth = 2.5
  ctx.lineCap = 'round'
  ctx.shadowColor = color
  ctx.shadowBlur = 8
  const corners: Array<[number, number, number, number]> = [
    [x, y, 1, 1],
    [x + w, y, -1, 1],
    [x, y + h, 1, -1],
    [x + w, y + h, -1, -1],
  ]
  for (const [cx, cy, dx, dy] of corners) {
    ctx.beginPath()
    ctx.moveTo(cx + dx * len, cy)
    ctx.lineTo(cx, cy)
    ctx.lineTo(cx, cy + dy * len)
    ctx.stroke()
  }
  ctx.restore()
}

/** Resalta cada ojo con una elipse translúcida cuando el backend reporta cierre. */
function drawEyeClosure(
  ctx: CanvasRenderingContext2D,
  points: Array<[number, number, number]>,
  width: number,
  height: number,
): void {
  ctx.save()
  ctx.fillStyle = EYE_CLOSED_FILL
  ctx.strokeStyle = EYE_CLOSED_STROKE
  ctx.lineWidth = 1.5
  for (const indices of [LEFT_EYE, RIGHT_EYE]) {
    const region = regionOf(points, indices)
    if (region === null) continue
    const rx = Math.max(region.rx * width, 4)
    const ry = Math.max(region.ry * height, 2.5)
    ctx.beginPath()
    ctx.ellipse(region.cx * width, region.cy * height, rx, ry, 0, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
  }
  ctx.restore()
}

/** Resalta la boca con una elipse translúcida cuando el backend reporta bostezo. */
function drawMouthHighlight(
  ctx: CanvasRenderingContext2D,
  points: Array<[number, number, number]>,
  width: number,
  height: number,
): void {
  const region = regionOf(points, MOUTH_REGION)
  if (region === null) return
  const rx = Math.max(region.rx * width, 8)
  const ry = Math.max(region.ry * height, 3)
  ctx.save()
  ctx.fillStyle = MOUTH_FILL
  ctx.strokeStyle = MOUTH_STROKE
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.ellipse(region.cx * width, region.cy * height, rx, ry, 0, 0, Math.PI * 2)
  ctx.fill()
  ctx.stroke()
  ctx.restore()
}

/** Flecha con cabeza triangular desde (x0, y0) a lo largo de (dx, dy). */
function drawArrow(
  ctx: CanvasRenderingContext2D,
  x0: number,
  y0: number,
  dx: number,
  dy: number,
  lengthPx: number,
): void {
  const magnitude = Math.hypot(dx, dy)
  if (magnitude < 1e-6) return
  const ux = dx / magnitude
  const uy = dy / magnitude
  const x1 = x0 + ux * lengthPx
  const y1 = y0 + uy * lengthPx
  ctx.beginPath()
  ctx.moveTo(x0, y0)
  ctx.lineTo(x1, y1)
  ctx.stroke()
  const angle = Math.atan2(uy, ux)
  const head = 7
  ctx.beginPath()
  ctx.moveTo(x1, y1)
  ctx.lineTo(x1 - head * Math.cos(angle - Math.PI / 6), y1 - head * Math.sin(angle - Math.PI / 6))
  ctx.lineTo(x1 - head * Math.cos(angle + Math.PI / 6), y1 - head * Math.sin(angle + Math.PI / 6))
  ctx.closePath()
  ctx.fill()
}

/**
 * Dirección de la mirada: una flecha por anillo de iris a lo largo de la
 * dirección normalizada que provee el backend, más el dot de nariz como
 * referencia estable del rostro (como en S1-15).
 *
 * Vista en espejo (S1-18 UX): se invierte SOLO el eje X para que la flecha
 * siga la dirección percibida por el usuario; el eje Y queda intacto.
 */
function drawGaze(
  ctx: CanvasRenderingContext2D,
  face: FaceOverlayFace,
  isMirrored: boolean,
  width: number,
  height: number,
): void {
  const nose = face.points[NOSE_INDEX]
  if (nose !== undefined) {
    const [nx, ny] = nose
    ctx.fillStyle = NOSE_COLOR
    ctx.beginPath()
    ctx.arc(nx * width, ny * height, 3, 0, Math.PI * 2)
    ctx.fill()
  }
  if (face.iris_direction === undefined) return
  const { x, y } = face.iris_direction
  const gx = isMirrored ? -x : x
  ctx.save()
  ctx.strokeStyle = GAZE_COLOR
  ctx.fillStyle = GAZE_COLOR
  ctx.lineWidth = 2
  ctx.lineCap = 'round'
  for (const indices of [LEFT_IRIS, RIGHT_IRIS]) {
    const centroid = ringCentroid(face.points, indices)
    if (centroid === null) continue
    const [ex, ey] = centroid
    drawArrow(ctx, ex * width, ey * height, gx, y, GAZE_ARROW_PX)
  }
  ctx.restore()
}

/**
 * Gizmo de orientación de cabeza (yaw/pitch/roll) a la derecha de la cara:
 * círculo con crosshair rotado por roll y un dot desplazado por yaw/pitch.
 * La conversión radianes->grados es matemática de presentación, permitida.
 *
 * Vista en espejo (S1-18 UX): se invierte SOLO el yaw (giro horizontal) para
 * que el indicador siga la orientación percibida por el usuario; pitch y roll
 * quedan intactos porque el espejo no altera la inclinación ni el balanceo.
 */
function drawHeadPoseGizmo(
  ctx: CanvasRenderingContext2D,
  bbox: [number, number, number, number],
  headPose: HeadPose,
  isMirrored: boolean,
  width: number,
  height: number,
): void {
  const [bx, by, bw] = bbox
  const gx = (bx + bw) * width + 18
  const gy = by * height + 14
  const radius = 20
  const yaw = ((isMirrored ? -headPose.yaw : headPose.yaw) * 180) / Math.PI
  const pitch = (headPose.pitch * 180) / Math.PI
  const roll = (headPose.roll * 180) / Math.PI

  ctx.save()
  ctx.strokeStyle = HEAD_POSE_COLOR
  ctx.lineWidth = 1.5
  ctx.beginPath()
  ctx.arc(gx, gy, radius, 0, Math.PI * 2)
  ctx.stroke()

  // Crosshair: la cruz se rota con el roll de la cabeza.
  ctx.save()
  ctx.translate(gx, gy)
  ctx.rotate((roll * Math.PI) / 180)
  ctx.beginPath()
  ctx.moveTo(-radius * 0.7, 0)
  ctx.lineTo(radius * 0.7, 0)
  ctx.moveTo(0, -radius * 0.7)
  ctx.lineTo(0, radius * 0.7)
  ctx.stroke()
  ctx.restore()

  // Dot de yaw/pitch: desplazamiento horizontal/vertical desde el centro.
  const dx = clamp(yaw / 90, -1, 1) * radius * 0.75
  const dy = clamp(pitch / 90, -1, 1) * radius * 0.75
  ctx.fillStyle = HEAD_POSE_COLOR
  ctx.beginPath()
  ctx.arc(gx + dx, gy + dy, 3.5, 0, Math.PI * 2)
  ctx.fill()

  // Etiqueta Y/P/R con los valores en grados.
  ctx.font = '10px ui-monospace, monospace'
  ctx.textAlign = 'center'
  ctx.fillStyle = HUD_TEXT_COLOR
  ctx.fillText(`Y ${fmtDeg(yaw)}°  P ${fmtDeg(pitch)}°  R ${fmtDeg(roll)}°`, gx, gy + radius + 14)
  ctx.restore()
}

/**
 * Anillo de estado alrededor del bbox, coloreado por el estado de riesgo que
 * llega como prop (assessment.state ya mapeado en App.tsx). SOLO presentación:
 * no clasifica nada acá.
 */
function drawStatusRing(
  ctx: CanvasRenderingContext2D,
  bbox: [number, number, number, number],
  state: StatusMeta,
  width: number,
  height: number,
): void {
  const [bx, by, bw, bh] = bbox
  const x = bx * width
  const y = by * height
  const w = bw * width
  const h = bh * height
  const style = STATUS_STYLE[state.code] ?? DEFAULT_STATUS_STYLE
  const pad = 7

  ctx.save()
  ctx.strokeStyle = style.color
  ctx.lineWidth = 3
  ctx.beginPath()
  ctx.roundRect(x - pad, y - pad, w + pad * 2, h + pad * 2, 12)
  ctx.stroke()
  ctx.restore()

  const label = `${style.emoji} ${state.label}`
  ctx.save()
  ctx.font = 'bold 12px ui-sans-serif, system-ui, sans-serif'
  const textWidth = ctx.measureText(label).width
  const pillX = x + w / 2 - textWidth / 2 - 8
  const pillY = y - pad - 26
  ctx.fillStyle = 'rgba(2, 6, 23, 0.6)'
  ctx.beginPath()
  ctx.roundRect(pillX, pillY, textWidth + 16, 20, 10)
  ctx.fill()
  ctx.fillStyle = style.color
  ctx.textAlign = 'center'
  ctx.fillText(label, pillX + (textWidth + 16) / 2, pillY + 14)
  ctx.restore()
}

/** HUD superior izquierdo: FPS, latencia y estado del WebSocket. */
function drawHud(ctx: CanvasRenderingContext2D, hud: HudData, t: T): void {
  const connection =
    hud.connection === 'connected'
      ? { color: '#22c55e', label: t('connected') }
      : hud.connection === 'connecting'
        ? { color: '#f59e0b', label: t('connecting') }
        : { color: '#ef4444', label: t('disconnected') }
  const lines: Array<{ text: string; dot?: string }> = [
    { text: `${t('hudFps')}: ${hud.fps}` },
    { text: `${t('hudLatency')}: ${hud.latencyMs.toFixed(1)} ms` },
    { text: `${t('hudConnection')}: ${connection.label}`, dot: connection.color },
  ]

  ctx.save()
  ctx.font = '12px ui-monospace, monospace'
  const lineHeight = 18
  const maxWidth = lines.reduce((max, line) => Math.max(max, ctx.measureText(line.text).width), 0)
  const padX = 14
  const padY = 10
  const boxX = 8
  const boxY = 8
  const boxW = maxWidth + padX * 2 + 16
  const boxH = lines.length * lineHeight + padY * 2
  ctx.fillStyle = HUD_BG_COLOR
  ctx.beginPath()
  ctx.roundRect(boxX, boxY, boxW, boxH, 8)
  ctx.fill()
  ctx.textBaseline = 'middle'
  ctx.fillStyle = HUD_TEXT_COLOR
  lines.forEach((line, index) => {
    const x = boxX + padX
    const y = boxY + padY + index * lineHeight + lineHeight / 2
    if (line.dot !== undefined) {
      ctx.fillStyle = line.dot
      ctx.fillText('●', x, y)
      ctx.fillStyle = HUD_TEXT_COLOR
    }
    ctx.fillText(line.text, x + (line.dot !== undefined ? 12 : 0), y)
  })
  ctx.restore()
}

function drawFace(
  ctx: CanvasRenderingContext2D,
  face: FaceOverlayFace,
  options: OverlayOptions,
  state: StatusMeta | null,
  isMirrored: boolean,
  width: number,
  height: number,
): void {
  const [bx, by, bw, bh] = face.bbox

  if (options.bbox) {
    drawCornerBrackets(ctx, bx * width, by * height, bw * width, bh * height, BBOX_COLOR)
  }

  if (options.landmarks) {
    ctx.fillStyle = MESH_COLOR
    for (const point of face.points) {
      const [px, py] = point
      ctx.beginPath()
      ctx.arc(px * width, py * height, 1.5, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  if (options.eyes && face.eyes_closed === true) {
    drawEyeClosure(ctx, face.points, width, height)
  }

  if (options.mouth && face.yawning === true) {
    drawMouthHighlight(ctx, face.points, width, height)
  }

  if (options.gaze) {
    drawGaze(ctx, face, isMirrored, width, height)
  }

  if (options.headPose && face.head_pose !== undefined) {
    drawHeadPoseGizmo(ctx, face.bbox, face.head_pose, isMirrored, width, height)
  }

  if (options.statusRing && state !== null) {
    drawStatusRing(ctx, face.bbox, state, width, height)
  }
}

function drawOverlay(
  ctx: CanvasRenderingContext2D,
  overlay: FaceOverlay | null,
  options: OverlayOptions,
  state: StatusMeta | null,
  isMirrored: boolean,
  hud: HudData,
  width: number,
  height: number,
  t: T,
): void {
  if (options.hud) {
    drawHud(ctx, hud, t)
  }
  if (overlay === null || overlay.face_count === 0) return
  for (const face of overlay.faces) {
    drawFace(ctx, face, options, state, isMirrored, width, height)
  }
}

/**
 * Video local (getUserMedia) con overlay de detección recibido del backend.
 * El canvas se redimensiona al contenedor (ResizeObserver) y el overlay, en
 * coordenadas normalizadas [0,1], se escala al rect del contenedor en cada
 * frame con requestAnimationFrame. No calcula NADA de negocio: solo dibuja
 * lo recibido y lo que App.tsx le pasa como props (toggles, estado de riesgo,
 * FPS/latencia/conexión para el HUD).
 */
export function WebcamFeed({
  stream,
  overlay,
  fps,
  latencyMs,
  connection,
  options,
  state = null,
  isMirrored = false,
  frameProvide = null,
}: WebcamFeedProps) {
  const { t } = useLanguage()
  const containerRef = useRef<HTMLDivElement | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const overlayRef = useRef<FaceOverlay | null>(null)
  const optionsRef = useRef<OverlayOptions>(options)
  const stateRef = useRef<StatusMeta | null>(state)
  const isMirroredRef = useRef(isMirrored)
  const hudRef = useRef<HudData>({ fps, latencyMs, connection })
  const tRef = useRef<T>(t)
  const sizeRef = useRef<Size>({ width: 0, height: 0, dpr: 1 })

  overlayRef.current = overlay
  optionsRef.current = options
  stateRef.current = state
  isMirroredRef.current = isMirrored
  hudRef.current = { fps, latencyMs, connection }
  tRef.current = t

  useBrowserFrameSender(videoRef, frameProvide)

  useEffect(() => {
    const video = videoRef.current
    if (video === null) return
    video.srcObject = stream
    return () => {
      if (video.srcObject === stream) video.srcObject = null
    }
  }, [stream])

  useEffect(() => {
    const container = containerRef.current
    const canvas = canvasRef.current
    if (container === null || canvas === null) return

    const resize = () => {
      const dpr = window.devicePixelRatio || 1
      const width = container.clientWidth
      const height = container.clientHeight
      sizeRef.current = { width, height, dpr }
      canvas.width = Math.max(1, Math.round(width * dpr))
      canvas.height = Math.max(1, Math.round(height * dpr))
    }
    resize()
    const observer = new ResizeObserver(resize)
    observer.observe(container)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (canvas === null) return
    const ctx = canvas.getContext('2d')
    if (ctx === null) return

    let rafId = 0
    const draw = () => {
      const { width, height, dpr } = sizeRef.current
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, width, height)
      if (width > 0 && height > 0) {
        drawOverlay(
          ctx,
          overlayRef.current,
          optionsRef.current,
          stateRef.current,
          isMirroredRef.current,
          hudRef.current,
          width,
          height,
          tRef.current,
        )
      }
      rafId = requestAnimationFrame(draw)
    }
    rafId = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(rafId)
  }, [])

  const noFace = overlay === null || overlay.face_count === 0

  return (
    <div
      ref={containerRef}
      className="relative aspect-[4/3] overflow-hidden rounded-xl border border-slate-800 bg-slate-950"
    >
      <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 h-full w-full object-cover" />
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
      {noFace ? (
        <div className="pointer-events-none absolute inset-x-0 top-3 flex justify-center">
          <span className="rounded-full bg-slate-900/80 px-3 py-1 text-xs font-medium text-slate-300">
            {t('noFaceDetected')}
          </span>
        </div>
      ) : null}
    </div>
  )
}