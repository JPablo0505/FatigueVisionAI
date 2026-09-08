import type { RiskState } from '@fatiguevision/contracts'

/**
 * Motor de sonido de alertas con Web Audio API (S1-19, S4-02).
 *
 * Soporta 4 tipos de alarma seleccionables por el usuario:
 * - beep:    Tono simple y limpio (original).
 * - siren:   Sirena ascendente/descendente estilo emergency vehicle.
 * - emergency: Tono harsh estilo EAS (Emergency Alert System).
 * - buzzer:  Zumbador continuo y penetrante.
 *
 * NO usa archivos de audio: todo se genera con osciladores Web Audio.
 */

export type AlarmType = 'beep' | 'siren' | 'emergency' | 'buzzer'

export const ALARM_TYPES: Array<{ id: AlarmType; labelKey: string }> = [
  { id: 'beep', labelKey: 'alarmBeep' },
  { id: 'siren', labelKey: 'alarmSiren' },
  { id: 'emergency', labelKey: 'alarmEmergency' },
  { id: 'buzzer', labelKey: 'alarmBuzzer' },
]

/** Handle de un sonido en curso: permite silenciarlo antes de que termine. */
export interface AlertSoundHandle {
  stop: () => void
}

interface ToneSpec {
  /** Frecuencia del tono en Hz. */
  frequency: number
  /** Duración del tono en segundos. */
  duration: number
  /** Pausa después del tono, en segundos. */
  gap: number
  /** Ganancia pico (0..1). */
  gain: number
  /** Tipo de oscilador (default: sine). */
  type?: OscillatorType
}

interface SoundPattern {
  /** Secuencia de tonos de un ciclo. */
  tones: ToneSpec[]
  /** Duración total del patrón repetido; 0 = una sola pasada. */
  repeatMs: number
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PATRONES POR TIPO DE ALARMA
 * ═══════════════════════════════════════════════════════════════════════════ */

const BEEP_PATTERNS: Partial<Record<RiskState, SoundPattern>> = {
  fatigue_warning: {
    tones: [{ frequency: 880, duration: 0.15, gap: 0, gain: 0.15 }],
    repeatMs: 0,
  },
  microsleep_suspected: {
    tones: [
      { frequency: 988, duration: 0.2, gap: 0.15, gain: 0.25 },
      { frequency: 988, duration: 0.2, gap: 0, gain: 0.25 },
    ],
    repeatMs: 0,
  },
  critical_microsleep: {
    tones: [
      { frequency: 880, duration: 0.12, gap: 0.08, gain: 0.45 },
      { frequency: 1100, duration: 0.12, gap: 0.08, gain: 0.5 },
      { frequency: 880, duration: 0.12, gap: 0.3, gain: 0.45 },
    ],
    repeatMs: 5_000,
  },
}

const SIREN_PATTERNS: Partial<Record<RiskState, SoundPattern>> = {
  fatigue_warning: {
    tones: [
      { frequency: 600, duration: 0.3, gap: 0, gain: 0.2 },
      { frequency: 900, duration: 0.3, gap: 0, gain: 0.2 },
    ],
    repeatMs: 0,
  },
  microsleep_suspected: {
    tones: [
      { frequency: 500, duration: 0.4, gap: 0, gain: 0.35 },
      { frequency: 1000, duration: 0.4, gap: 0, gain: 0.35 },
      { frequency: 500, duration: 0.4, gap: 0, gain: 0.35 },
    ],
    repeatMs: 0,
  },
  critical_microsleep: {
    tones: [
      { frequency: 400, duration: 0.5, gap: 0, gain: 0.5 },
      { frequency: 1200, duration: 0.5, gap: 0, gain: 0.5 },
      { frequency: 400, duration: 0.5, gap: 0, gain: 0.5 },
      { frequency: 1200, duration: 0.5, gap: 0, gain: 0.5 },
    ],
    repeatMs: 5_000,
  },
}

const EMERGENCY_PATTERNS: Partial<Record<RiskState, SoundPattern>> = {
  fatigue_warning: {
    tones: [
      { frequency: 800, duration: 0.1, gap: 0.05, gain: 0.2, type: 'square' },
      { frequency: 1000, duration: 0.1, gap: 0.05, gain: 0.2, type: 'square' },
    ],
    repeatMs: 0,
  },
  microsleep_suspected: {
    tones: [
      { frequency: 800, duration: 0.08, gap: 0.04, gain: 0.35, type: 'square' },
      { frequency: 1000, duration: 0.08, gap: 0.04, gain: 0.35, type: 'square' },
      { frequency: 800, duration: 0.08, gap: 0.04, gain: 0.35, type: 'square' },
      { frequency: 1000, duration: 0.08, gap: 0.04, gain: 0.35, type: 'square' },
    ],
    repeatMs: 0,
  },
  critical_microsleep: {
    tones: [
      { frequency: 800, duration: 0.2, gap: 0.05, gain: 0.55, type: 'sawtooth' },
      { frequency: 1000, duration: 0.2, gap: 0.05, gain: 0.55, type: 'sawtooth' },
      { frequency: 600, duration: 0.15, gap: 0.05, gain: 0.55, type: 'sawtooth' },
      { frequency: 1200, duration: 0.15, gap: 0.05, gain: 0.55, type: 'sawtooth' },
      { frequency: 800, duration: 0.15, gap: 0.4, gain: 0.55, type: 'sawtooth' },
    ],
    repeatMs: 5_000,
  },
}

const BUZZER_PATTERNS: Partial<Record<RiskState, SoundPattern>> = {
  fatigue_warning: {
    tones: [
      { frequency: 200, duration: 0.3, gap: 0.1, gain: 0.2, type: 'sawtooth' },
    ],
    repeatMs: 0,
  },
  microsleep_suspected: {
    tones: [
      { frequency: 200, duration: 0.4, gap: 0.1, gain: 0.35, type: 'sawtooth' },
      { frequency: 250, duration: 0.4, gap: 0.1, gain: 0.35, type: 'sawtooth' },
    ],
    repeatMs: 0,
  },
  critical_microsleep: {
    tones: [
      { frequency: 180, duration: 0.8, gap: 0.1, gain: 0.6, type: 'sawtooth' },
      { frequency: 220, duration: 0.8, gap: 0.1, gain: 0.6, type: 'sawtooth' },
      { frequency: 180, duration: 0.8, gap: 0.1, gain: 0.6, type: 'sawtooth' },
    ],
    repeatMs: 5_000,
  },
}

const ALL_PATTERNS: Record<AlarmType, Partial<Record<RiskState, SoundPattern>>> = {
  beep: BEEP_PATTERNS,
  siren: SIREN_PATTERNS,
  emergency: EMERGENCY_PATTERNS,
  buzzer: BUZZER_PATTERNS,
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ENGINE
 * ═══════════════════════════════════════════════════════════════════════════ */

/** Ataque/decaimiento suaves para evitar clics al cortar tonos. */
const ATTACK_SECONDS = 0.01
const RELEASE_SECONDS = 0.02
/** Retardo inicial antes del primer tono. */
const LEAD_SECONDS = 0.02

let enabled = true
let volumePercent = 80
let alarmType: AlarmType = 'siren'
let context: AudioContext | null = null
let activeHandle: AlertSoundHandle | null = null

function windowAudioContext(): (typeof AudioContext) | null {
  if (typeof window === 'undefined') return null
  if (typeof window.AudioContext !== 'undefined') return window.AudioContext
  const webkit = (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
  return webkit ?? null
}

function ensureContext(): AudioContext | null {
  const Ctor = windowAudioContext()
  if (Ctor === null) return null
  if (context === null) {
    try {
      context = new Ctor()
    } catch {
      return null
    }
  }
  return context
}

/** Reanuda el contexto si está suspendido; si el navegador lo bloquea, no pasa nada. */
function resumeContext(): void {
  const ctx = context
  if (ctx === null || ctx.state !== 'suspended') return
  void ctx.resume().catch(() => {})
}

function stopActive(): void {
  if (activeHandle !== null) {
    activeHandle.stop()
    activeHandle = null
  }
}

/** Programa los tonos del patrón y devuelve un handle para cortarlos antes. */
function scheduleCycle(ctx: AudioContext, pattern: SoundPattern): AlertSoundHandle {
  const master = ctx.createGain()
  master.gain.value = volumePercent / 100
  master.connect(ctx.destination)
  const scheduled: Array<{ osc: OscillatorNode; gain: GainNode }> = []

  const cycleSeconds = pattern.tones.reduce((sum, tone) => sum + tone.duration + tone.gap, 0)
  const repeats = pattern.repeatMs > 0 ? Math.max(1, Math.floor(pattern.repeatMs / 1000 / cycleSeconds)) : 1

  let cursor = ctx.currentTime + LEAD_SECONDS
  for (let r = 0; r < repeats; r++) {
    for (const tone of pattern.tones) {
      const osc = ctx.createOscillator()
      osc.type = tone.type ?? 'sine'
      osc.frequency.value = tone.frequency
      const gain = ctx.createGain()
      gain.gain.setValueAtTime(0.0001, cursor)
      gain.gain.linearRampToValueAtTime(tone.gain, cursor + ATTACK_SECONDS)
      gain.gain.setValueAtTime(tone.gain, cursor + tone.duration - RELEASE_SECONDS)
      gain.gain.linearRampToValueAtTime(0.0001, cursor + tone.duration)
      osc.connect(gain)
      gain.connect(master)
      osc.start(cursor)
      osc.stop(cursor + tone.duration + RELEASE_SECONDS)
      scheduled.push({ osc, gain })
      cursor += tone.duration + tone.gap
    }
  }

  let stopped = false
  const handle: AlertSoundHandle = {
    stop() {
      if (stopped) return
      stopped = true
      const now = ctx.currentTime
      master.gain.cancelScheduledValues(now)
      master.gain.setValueAtTime(0.0001, now)
      for (const { osc, gain } of scheduled) {
        try {
          osc.stop(now)
        } catch {
          // Ya terminó por sí solo; ignorar.
        }
        gain.disconnect()
      }
      master.disconnect()
    },
  }
  return handle
}

export const alertSound = {
  /** Activa/desactiva el sonido. El toggle es un gesto de usuario que desbloquea el audio. */
  setEnabled(value: boolean): void {
    enabled = value
    if (!value) {
      stopActive()
    } else {
      const ctx = ensureContext()
      if (ctx !== null) resumeContext()
    }
  },

  /** Ajusta el volumen maestro (0–100). */
  setVolume(percent: number): void {
    volumePercent = Math.max(0, Math.min(100, percent))
  },

  /** Selecciona el tipo de alarma. */
  setType(type: AlarmType): void {
    alarmType = type
  },

  /** Obtiene el tipo actual. */
  getType(): AlarmType {
    return alarmType
  },

  /**
   * Reproduce el patrón del estado dado con el tipo de alarma seleccionado.
   * Devuelve un handle para silenciar, o null si está mutado / no aplica / el contexto no existe.
   */
  play(state: RiskState): AlertSoundHandle | null {
    if (!enabled) return null
    const patterns = ALL_PATTERNS[alarmType]
    const pattern = patterns?.[state]
    if (pattern === undefined || pattern.tones.length === 0) return null
    const ctx = ensureContext()
    if (ctx === null) return null
    if (ctx.state === 'suspended') {
      resumeContext()
    }
    stopActive()
    activeHandle = scheduleCycle(ctx, pattern)
    return activeHandle
  },

  /** Cierra el contexto de audio (p. ej. al desmontar la app). */
  dispose(): void {
    stopActive()
    if (context !== null) {
      void context.close().catch(() => {})
      context = null
    }
  },
}
