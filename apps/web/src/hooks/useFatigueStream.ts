import { useCallback, useEffect, useRef, useState } from 'react'
import { MessageKind, WS_PATH } from '@fatiguevision/contracts'
import type {
  BiometricEvent,
  ClientToServerMessage,
  ErrorPayload,
  FaceOverlay,
  FatigueAssessment,
  VisionStatus,
} from '@fatiguevision/contracts'
import { buildPing } from '../lib/envelope'
import { parseServerMessage } from '../lib/messages'

/**
 * Estados de conexión del WebSocket, según ReadyState del navegador.
 * `closed` con `reconnectAttempt > 0` significa "reconectando".
 */
export type ConnectionState = 'connecting' | 'open' | 'closing' | 'closed'

const PING_INTERVAL_MS = 5_000
const RECONNECT_BASE_MS = 1_000
const RECONNECT_MAX_MS = 10_000
/** Límite del buffer de eventos biométricos recientes (los más viejos se descartan). */
const MAX_RECENT_EVENTS = 50

/** Backoff exponencial simple: 1s -> 2s -> 4s -> 8s -> 10s (tope). */
function reconnectDelayMs(attempt: number): number {
  return Math.min(RECONNECT_BASE_MS * 2 ** attempt, RECONNECT_MAX_MS)
}

export interface FatigueStream {
  connectionState: ConnectionState
  reconnectAttempt: number
  visionStatus: VisionStatus | null
  assessment: FatigueAssessment | null
  overlay: FaceOverlay | null
  serverError: ErrorPayload | null
  lastPongAt: number | null
  events: BiometricEvent[]
  send: (message: ClientToServerMessage) => boolean
}

/**
 * Hook del stream de fatiga: conecta al WS del backend, reconecta con
 * backoff, hace ping periódico y despacha por `kind`. Solo visualiza/
 * reenvía lo que manda el backend; no calcula nada.
 */
export function useFatigueStream(): FatigueStream {
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting')
  const [reconnectAttempt, setReconnectAttempt] = useState(0)
  const [visionStatus, setVisionStatus] = useState<VisionStatus | null>(null)
  const [assessment, setAssessment] = useState<FatigueAssessment | null>(null)
  const [overlay, setOverlay] = useState<FaceOverlay | null>(null)
  const [serverError, setServerError] = useState<ErrorPayload | null>(null)
  const [lastPongAt, setLastPongAt] = useState<number | null>(null)
  const [events, setEvents] = useState<BiometricEvent[]>([])

  const socketRef = useRef<WebSocket | null>(null)
  const reconnectTimerRef = useRef<number | null>(null)
  const pingTimerRef = useRef<number | null>(null)

  useEffect(() => {
    let disposed = false
    let attempt = 0

    const clearTimers = () => {
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }
      if (pingTimerRef.current !== null) {
        window.clearInterval(pingTimerRef.current)
        pingTimerRef.current = null
      }
    }

    const teardown = () => {
      disposed = true
      clearTimers()
      const ws = socketRef.current
      socketRef.current = null
      if (ws !== null) {
        ws.onopen = null
        ws.onmessage = null
        ws.onclose = null
        ws.close()
      }
    }

    // Conexión directa al backend (S3-02): bypass Vite proxy para evitar
    // ECONNABORTED por alto volumen de frames.  VITE_WS_URL permite
    // override completo; sin ella, se construye la URL absoluta apuntando
    // al mismo host en el puerto del backend (8000).
    const connect = () => {
      if (disposed) return
      setConnectionState('connecting')
      const wsUrl =
        import.meta.env.VITE_WS_URL ||
        `ws://${window.location.hostname}:8000${WS_PATH}`
      const ws = new WebSocket(wsUrl)
      socketRef.current = ws

      ws.onopen = () => {
        if (disposed || socketRef.current !== ws) return
        attempt = 0
        setReconnectAttempt(0)
        setConnectionState('open')
      }

      ws.onmessage = (event: MessageEvent) => {
        if (disposed || socketRef.current !== ws) return
        const message = parseServerMessage(event.data)
        if (message === null) return
        switch (message.kind) {
          case MessageKind.VisionStatus:
            setVisionStatus(message.payload)
            break
          case MessageKind.FatigueAssessment:
            setAssessment(message.payload)
            break
          case MessageKind.FaceOverlay:
            setOverlay(message.payload)
            break
          case MessageKind.BiometricEvent:
            // Buffer cronológico, tope de MAX_RECENT_EVENTS (se descartan los
            // más viejos); el panel muestra el orden invertido (nuevo primero).
            setEvents((prev) => [...prev.slice(-(MAX_RECENT_EVENTS - 1)), message.payload])
            break
          case MessageKind.Error:
            setServerError(message.payload)
            break
          case MessageKind.Pong:
            setLastPongAt(message.timestamp)
            break
        }
      }

      ws.onclose = () => {
        if (disposed || socketRef.current !== ws) return
        socketRef.current = null
        setConnectionState('closed')
        attempt += 1
        setReconnectAttempt(attempt)
        reconnectTimerRef.current = window.setTimeout(connect, reconnectDelayMs(attempt))
      }

      ws.onerror = () => {
        // el cierre llega por onclose; acá solo dejamos el estado que maneja onclose
      }
    }

    connect()

    return teardown
  }, [])

  // Ping periódico de keep-alive mientras la conexión esté abierta.
  useEffect(() => {
    if (connectionState !== 'open') return
    pingTimerRef.current = window.setInterval(() => {
      const ws = socketRef.current
      if (ws !== null && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(buildPing()))
      }
    }, PING_INTERVAL_MS)
    return () => {
      if (pingTimerRef.current !== null) {
        window.clearInterval(pingTimerRef.current)
        pingTimerRef.current = null
      }
    }
  }, [connectionState])

  const send = useCallback((message: ClientToServerMessage): boolean => {
    const ws = socketRef.current
    if (ws === null || ws.readyState !== WebSocket.OPEN) return false
    ws.send(JSON.stringify(message))
    return true
  }, [])

  return {
    connectionState,
    reconnectAttempt,
    visionStatus,
    assessment,
    overlay,
    serverError,
    lastPongAt,
    events,
    send,
  }
}