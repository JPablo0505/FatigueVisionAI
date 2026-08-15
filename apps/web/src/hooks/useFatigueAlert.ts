import { useEffect, useRef, useState } from 'react'
import type { FatigueAssessment, RiskState } from '@fatiguevision/contracts'
import { alertSound } from '../lib/alertSound'
import type { AlertSoundHandle } from '../lib/alertSound'

/** Contenido de la alerta temporal: estado + id para forzar remount al reemplazar. */
export interface FatigueAlertInfo {
  state: RiskState
  id: number
}

/** Orden de severidad ascendente (0 = normal). */
const SEVERITY: Record<RiskState, number> = {
  normal: 0,
  long_blink: 1,
  fatigue_warning: 2,
  microsleep_suspected: 3,
  critical_microsleep: 4,
}

/** Duración de una alerta temporal antes de desvanecerse (cuando no persiste). */
const ALERT_DURATION_MS = 6_000

function riskStateOf(assessment: FatigueAssessment | null): RiskState {
  return assessment?.state ?? 'normal'
}

function eyeClosureDuration(assessment: FatigueAssessment | null): number {
  return assessment?.features?.current_eye_closure_duration ?? 0
}

/**
 * Hook de alertas temporales de fatiga (S1-19).
 *
 * Separa la alerta PERSISTENTE (RiskBanner sigue mostrando el estado actual)
 * de las notificaciones TEMPORALES: solo una transición ascendente de severidad
 * dispara una alerta (y su sonido). La alerta crítica con los ojos aún cerrados
 * no expira mientras dure el microsueño.
 */
export function useFatigueAlert(
  assessment: FatigueAssessment | null,
  soundEnabled: boolean,
): { alert: FatigueAlertInfo | null; dismiss: () => void } {
  const [alert, setAlert] = useState<FatigueAlertInfo | null>(null)
  // Línea base: `null` hasta recibir el primer assessment. El primer estado
  // solo SE ESTABLECE, nunca dispara alerta/sonido: un crítico de arranque
  // (ojos cerrados al prender, cámara iniciando) no es una transición real.
  const prevStateRef = useRef<RiskState | null>(null)
  const alertRef = useRef<FatigueAlertInfo | null>(null)
  const expiryTimerRef = useRef<number | null>(null)
  const soundHandleRef = useRef<AlertSoundHandle | null>(null)
  const soundEnabledRef = useRef(soundEnabled)
  const idRef = useRef(0)

  // Mantener la preferencia más reciente sin depender del closure del efecto.
  useEffect(() => {
    soundEnabledRef.current = soundEnabled
  }, [soundEnabled])

  // Sincronizar el motor de audio con la preferencia; el click del toggle es
  // el gesto de usuario que desbloquea el AudioContext.
  useEffect(() => {
    alertSound.setEnabled(soundEnabled)
  }, [soundEnabled])

  const clearExpiry = () => {
    if (expiryTimerRef.current !== null) {
      window.clearTimeout(expiryTimerRef.current)
      expiryTimerRef.current = null
    }
  }

  const startExpiry = () => {
    if (expiryTimerRef.current !== null) return
    expiryTimerRef.current = window.setTimeout(() => {
      expiryTimerRef.current = null
      stopSound()
      alertRef.current = null
      setAlert(null)
    }, ALERT_DURATION_MS)
  }

  const stopSound = () => {
    const handle = soundHandleRef.current
    if (handle !== null) {
      handle.stop()
      soundHandleRef.current = null
    }
  }

  const showAlert = (state: RiskState) => {
    const next: FatigueAlertInfo = { state, id: idRef.current++ }
    alertRef.current = next
    setAlert(next)
    // Expiración normal por defecto; se reinicia al reemplazar (transición
    // ascendente) y se cancela mientras dure un microsueño crítico activo.
    clearExpiry()
    startExpiry()
  }

  const dismiss = () => {
    stopSound()
    clearExpiry()
    alertRef.current = null
    setAlert(null)
  }

  useEffect(() => {
    const current = riskStateOf(assessment)
    const prev = prevStateRef.current
    prevStateRef.current = current

    // Primer assessment recibido: no es una transición, es el estado inicial
    // del stream. Sin esto, un primer estado alto (p. ej. crítico al prender)
    // dispararía sonido + alerta apenas conecta.
    if (prev === null) return

    const escalated = SEVERITY[current] > SEVERITY[prev]

    // Transición ascendente (nunca en el montaje inicial: prev arranca en
    // 'normal' y el primer assessment suele ser null). Dispara SONIDO +
    // alerta temporal.
    if (escalated && current !== 'normal') {
      if (soundEnabledRef.current) {
        stopSound()
        soundHandleRef.current = alertSound.play(current)
      }
      showAlert(current)
      return
    }

    // Microsueño crítico EN CURSO (ojos aún cerrados): la alerta NO expira y,
    // si se ocultó (dismiss o expiración previa), reaparece de inmediato.
    if (current === 'critical_microsleep' && eyeClosureDuration(assessment) > 0) {
      if (alertRef.current?.state !== 'critical_microsleep') {
        showAlert('critical_microsleep')
      } else {
        clearExpiry()
      }
      return
    }

    // Fuera de la persistencia crítica: asegurar que la expiración normal siga
    // activa (por si se había cancelado durante el cierre de ojos).
    if (alertRef.current?.state === 'critical_microsleep' && expiryTimerRef.current === null) {
      startExpiry()
    }
  }, [assessment])

  // Limpieza al desmontar: timers, sonido en curso y contexto de audio.
  useEffect(
    () => () => {
      clearExpiry()
      stopSound()
      alertSound.dispose()
    },
    [],
  )

  return { alert, dismiss }
}