import { MessageKind, PROTOCOL_VERSION } from '@fatiguevision/contracts'
import type { ControlPayload, FrameProvidePayload, MessageEnvelope, PingPayload } from '@fatiguevision/contracts'

/** Epoch en segundos (float), mismo canon que el resto del protocolo. */
export function nowInSeconds(): number {
  return Date.now() / 1000
}

/** ID único por mensaje saliente, para correlación client<->server. */
export function newMessageId(): string {
  return crypto.randomUUID()
}

/**
 * Construye un envelope del protocolo de forma segura por tipo: el `kind`
 * queda acoplado al tipo del payload, así no se puede armar un ping con el
 * payload de un control sin error de compilación.
 */
export function buildEnvelope<K extends MessageKind, P>(kind: K, payload: P): MessageEnvelope<K, P> {
  return {
    messageId: newMessageId(),
    timestamp: nowInSeconds(),
    protocolVersion: PROTOCOL_VERSION,
    kind,
    payload,
  }
}

/** Ping de keep-alive (client -> server). */
export function buildPing(sentAt: number = nowInSeconds()): MessageEnvelope<typeof MessageKind.Ping, PingPayload> {
  return buildEnvelope(MessageKind.Ping, { sentAt })
}

/** Orden de arrancar el pipeline (client -> server). */
export function buildControlStart(): MessageEnvelope<typeof MessageKind.ControlStart, ControlPayload> {
  return buildEnvelope(MessageKind.ControlStart, {})
}

/** Orden de detener el pipeline (client -> server). */
export function buildControlStop(): MessageEnvelope<typeof MessageKind.ControlStop, ControlPayload> {
  return buildEnvelope(MessageKind.ControlStop, {})
}

/** Frame de cámara del navegador (client -> server), modo camera.source == "browser". */
export function buildFrameProvide(
  payload: FrameProvidePayload,
): MessageEnvelope<typeof MessageKind.FrameProvide, FrameProvidePayload> {
  return buildEnvelope(MessageKind.FrameProvide, payload)
}