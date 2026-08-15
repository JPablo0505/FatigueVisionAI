import { useEffect, useRef } from 'react'
import type { ClientToServerMessage } from '@fatiguevision/contracts'
import { buildFrameProvide } from '../lib/envelope'

const FRAME_WIDTH = 640
const FRAME_HEIGHT = 480
const JPEG_QUALITY = 0.7
const JPEG_DATA_URL_PREFIX = 'data:image/jpeg;base64,'
const FRAME_INTERVAL_MS = 50

/**
 * Destino de frames del navegador: `active` arranca/detiene el loop y `send`
 * entrega cada mensaje ya armado al WebSocket (la propia send restringe a ws OPEN).
 */
export interface FrameProvideSink {
  active: boolean
  send: (message: ClientToServerMessage) => boolean
}

/**
 * Sube frames del video local al backend en modo camera.source == "browser".
 *
 * Mientras `sink` sea no-nulo y esté activo, captura el frame actual del
 * video hacia un canvas 640x480, lo codifica como JPEG (calidad 0.7) y lo
 * envía como frame.provide. El loop corre con setInterval (~20 fps), se
 * detiene al desactivar el sink o al desmontar el componente, y descarta el
 * frame si el video todavía no está listo. Cada captura es una operación
 * corta que no bloquea el hilo principal.
 */
export function useBrowserFrameSender(
  videoRef: { current: HTMLVideoElement | null },
  sink: FrameProvideSink | null,
): void {
  const sinkRef = useRef(sink)
  sinkRef.current = sink

  const frameIdRef = useRef(0)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    if (sink === null || !sink.active) return

    const timer = window.setInterval(() => {
      const video = videoRef.current
      if (video === null || video.videoWidth === 0 || video.videoHeight === 0) return

      let canvas = canvasRef.current
      if (canvas === null) {
        canvas = document.createElement('canvas')
        canvas.width = FRAME_WIDTH
        canvas.height = FRAME_HEIGHT
        canvasRef.current = canvas
      }
      const ctx = canvas.getContext('2d')
      if (ctx === null) return

      ctx.drawImage(video, 0, 0, FRAME_WIDTH, FRAME_HEIGHT)
      const dataUrl = canvas.toDataURL('image/jpeg', JPEG_QUALITY)

      const payload = {
        frameId: frameIdRef.current,
        timestamp: Date.now() / 1000,
        width: FRAME_WIDTH,
        height: FRAME_HEIGHT,
        jpegBase64: dataUrl.slice(JPEG_DATA_URL_PREFIX.length),
      }
      frameIdRef.current += 1
      sinkRef.current?.send(buildFrameProvide(payload))
    }, FRAME_INTERVAL_MS)

    return () => window.clearInterval(timer)
  }, [sink, videoRef])

  // El frameId se reinicia por sesión: cada arranque del loop vuelve a 0 y
  // después crece de forma monótona hasta que el sink se desactiva.
  useEffect(() => {
    if (sink !== null && sink.active) {
      frameIdRef.current = 0
    }
  }, [sink])
}