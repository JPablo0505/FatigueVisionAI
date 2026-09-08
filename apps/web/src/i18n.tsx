import { createContext, useCallback, useContext, useMemo, useState } from 'react'
import type { ReactNode } from 'react'

export type Lang = 'en' | 'es'

export const LANG_STORAGE_KEY = 'fatiguevision.lang'

const DICTIONARY = {
  en: {
    // Header
    subtitle: 'Realtime fatigue monitoring',
    protocolBadge: 'Protocol version',
    protocolText: 'protocol {v}',
    connecting: 'Connecting...',
    connected: 'Connected',
    closing: 'Closing...',
    disconnected: 'Disconnected',
    reconnecting: 'Reconnecting...',
    attemptCount: 'attempt {n}',
    running: 'Running',
    stopped: 'Stopped',
    errorState: 'Error',
    pipeline: 'pipeline',
    na: 'n/a',
    lastUpdate: 'last update {t}',
    lastUpdateNone: 'last update —',
    languageSelector: 'Language',
    // Alerts
    connectionErrorTitle: 'Connection error',
    connectionErrorBody:
      'The dashboard is not connected to the backend. Retrying automatically...',
    serverErrorTitle: 'Server error ({code})',
    pipelineErrorTitle: 'Pipeline error',
    unknownPipelineError: 'Unknown pipeline error',
    // Controls
    controlsTitle: 'Controls',
    start: 'Start',
    stop: 'Stop',
    ping: 'Ping',
    pingKeepAlive: 'Ping keep-alive every ~5s',
    lastPong: ' · last pong {t}',
    // Camera placeholder
    cameraUnavailable: 'Camera unavailable',
    cameraPlaceholderHint: 'Enable camera access to see the live feed.',
    // System Status
    systemStatusTitle: 'System Status',
    waitingPipelineStatus: 'Waiting for pipeline status...',
    faceDetected: 'Face Detected',
    no: 'No',
    facesDetected: '{n} detected',
    streamFps: 'Stream FPS',
    latency: 'Latency',
    camera: 'Camera',
    runningVerb: 'running',
    idleVerb: 'idle',
    // Biometrics
    biometricsTitle: 'Biometrics',
    waitingAssessment: 'Waiting for assessment...',
    earLabel: 'EAR',
    earDesc: 'Eye Aspect Ratio — higher values mean more open eyes. Drops near zero during blinks or closure.',
    marLabel: 'MAR',
    marDesc: 'Mouth Aspect Ratio — higher values mean a wider mouth opening, typically from yawning.',
    avg: 'avg',
    perclosLabel: 'PERCLOS',
    perclosDesc: 'Percentage of time eyes were more than 80% closed. A key drowsiness indicator.',
    blinkRateLabel: 'Blink Rate',
    blinkRateDesc: 'Number of blinks per minute. Increased blink rate can signal fatigue.',
    eyeClosureLabel: 'Eye Closure',
    eyeClosureDesc: 'Average duration of eye closures in seconds.',
    currentYawnLabel: 'Current Yawn',
    currentYawnDesc: 'Duration of the yawning episode happening right now.',
    lastYawnLabel: 'Last Yawn',
    lastYawnDesc: 'Duration of the most recent completed yawn.',
    avgYawnLabel: 'Avg Yawn',
    avgYawnDesc: 'Average duration of all detected yawns in this session.',
    headPoseLabel: 'Head Pose',
    headPoseDesc: 'Head tilt in degrees — yaw (left/right), pitch (up/down), roll (tilt).',
    irisDirectionLabel: 'Iris Direction',
    irisDirectionDesc: 'Horizontal and vertical gaze offset of the iris center.',
    featuresMissing:
      'Biometric features not present in this assessment (older server).',
    earInstantLabel: 'Instant EAR',
    earInstantDesc: 'Real-time eye openness ratio, sampled every frame.',
    currentEyeClosureLabel: 'Current Closure',
    currentEyeClosureDesc: 'How long the eyes have been continuously closed right now.',
    longestEyeClosureLabel: 'Longest Closure',
    longestEyeClosureDesc: 'Longest single eye-closure event in this session — higher values indicate risk.',
    longestYawnLabel: 'Longest Yawn',
    longestYawnDesc: 'Duration of the longest yawn detected in this session.',
    totalBlinksLabel: 'Total Blinks',
    totalBlinksDesc: 'Running count of all blinks since the pipeline started.',
    totalYawnsLabel: 'Total Yawns',
    totalYawnsDesc: 'Running count of all yawns since the pipeline started.',
    // Risk state
    riskStateTitle: 'Risk State',
    riskStateAria: 'Realtime risk state',
    riskNormal: 'Normal',
    riskNormalDesc: 'No fatigue indicators detected.',
    riskLongBlink: 'Long Blinks',
    riskLongBlinkDesc: 'Closures lasting 0.3–0.5 s are appearing.',
    riskFatigueWarning: 'Fatigue Warning',
    riskFatigueWarningDesc: 'Sustained eye closure or other fatigue indicators.',
    riskMicrosleepSuspected: 'Microsleep Suspected',
    riskMicrosleepSuspectedDesc: 'A long eye closure consistent with microsleep was detected.',
    riskCriticalMicrosleep: 'Critical Microsleep',
    riskCriticalMicrosleepDesc: 'A very long eye closure was detected. Take immediate action.',
    // Recent events
    recentEventsTitle: 'Recent Events',
    emptyEvents: 'No completed biometric events yet.',
    eventBlink: 'Blink',
    eventLongBlink: 'Long blink',
    eventEyesClosed: 'Eyes closed',
    eventYawn: 'Yawn',
    // Fatigue
    fatigueTitle: 'Fatigue',
    waitingFatigue: 'Waiting for a fatigue assessment...',
    fatigueScore: 'Fatigue Score',
    fatigueScoreAria: 'Fatigue score',
    fatigueLevelAria: 'Fatigue level',
    levelLabel: 'level: {l}',
    levelLow: 'low',
    levelMedium: 'medium',
    levelHigh: 'high',
    confidence: 'Confidence',
    // Webcam feed
    noFaceDetected: 'No face detected',
    // Overlay controls (S1-18)
    overlayControlsTitle: 'Overlay Controls',
    overlayMirror: 'Mirror view (selfie)',
    alertSound: 'Alert sound',
    alertDismiss: 'Dismiss',
    overlayBbox: 'Bounding box',
    overlayLandmarks: 'Landmarks (468)',
    overlayEyes: 'Eyes closed highlight',
    overlayMouth: 'Yawn highlight',
    overlayGaze: 'Gaze direction',
    overlayHeadPose: 'Head pose gizmo',
    overlayStatusRing: 'Status ring',
    overlayHud: 'HUD (FPS / latency / WS)',
    hudFps: 'FPS',
    hudLatency: 'Latency',
    hudConnection: 'WS',
    // Settings (S4-02)
    settingsTitle: 'Settings',
    settingsClose: 'Close settings',
    settingsCameraDevice: 'Camera Device',
    settingsDefaultCamera: 'System default',
    settingsNoCameras: 'No cameras detected. Grant camera permission first.',
    settingsAlarmVolume: 'Alarm Volume',
    settingsModelMode: 'Detection Model',
    settingsModelModeDesc: 'The ML fatigue model runs server-side. Model type is configured via the backend environment variable FATIGUE_MODEL_TYPE.',
    settingsOverlays: 'Overlay Layers',
  },
  es: {
    // Header
    subtitle: 'Monitoreo de fatiga en tiempo real',
    protocolBadge: 'Versión del protocolo',
    protocolText: 'protocolo {v}',
    connecting: 'Conectando...',
    connected: 'Conectado',
    closing: 'Cerrando...',
    disconnected: 'Desconectado',
    reconnecting: 'Reconectando...',
    attemptCount: 'intento {n}',
    running: 'Corriendo',
    stopped: 'Detenido',
    errorState: 'Error',
    pipeline: 'pipeline',
    na: 'n/d',
    lastUpdate: 'última actualización {t}',
    lastUpdateNone: 'última actualización —',
    languageSelector: 'Idioma',
    // Alerts
    connectionErrorTitle: 'Error de conexión',
    connectionErrorBody:
      'El dashboard no está conectado al backend. Reintentando automáticamente...',
    serverErrorTitle: 'Error del servidor ({code})',
    pipelineErrorTitle: 'Error del pipeline',
    unknownPipelineError: 'Error de pipeline desconocido',
    // Controls
    controlsTitle: 'Controles',
    start: 'Iniciar',
    stop: 'Detener',
    ping: 'Ping',
    pingKeepAlive: 'Keep-alive de ping cada ~5 s',
    lastPong: ' · último pong {t}',
    // Camera placeholder
    cameraUnavailable: 'Cámara no disponible',
    cameraPlaceholderHint:
      'Habilita el acceso a la cámara para ver la transmisión en vivo.',
    // System Status
    systemStatusTitle: 'Estado del sistema',
    waitingPipelineStatus: 'Esperando el estado del pipeline...',
    faceDetected: 'Rostro detectado',
    no: 'No',
    facesDetected: '{n} detectados',
    streamFps: 'FPS del flujo',
    latency: 'Latencia',
    camera: 'Cámara',
    runningVerb: 'corriendo',
    idleVerb: 'inactivo',
    // Biometrics
    biometricsTitle: 'Biometría',
    waitingAssessment: 'Esperando evaluación...',
    earLabel: 'EAR (Relación de Apertura Ocular)',
    earDesc: 'Mide qué tan abiertos están los ojos. Valores altos = ojos bien abiertos; cercanos a cero durante parpadeos o cierres.',
    marLabel: 'MAR (Relación de Apertura Bucal)',
    marDesc: 'Mide qué tan abierta está la boca. Valores altos indican bostezos.',
    avg: 'prom',
    perclosLabel: 'PERCLOS',
    perclosDesc: 'Porcentaje de tiempo con los ojos cerrados más del 80%. Indicador clave de somnolencia.',
    blinkRateLabel: 'Parpadeos por minuto',
    blinkRateDesc: 'Cantidad de parpadeos por minuto. Un aumento puede indicar fatiga.',
    eyeClosureLabel: 'Cierre ocular',
    eyeClosureDesc: 'Duración promedio de los cierres de ojos en segundos.',
    currentYawnLabel: 'Bostezo actual',
    currentYawnDesc: 'Duración del bostezo que está ocurriendo ahora mismo.',
    lastYawnLabel: 'Último bostezo',
    lastYawnDesc: 'Duración del bostezo más reciente completado.',
    avgYawnLabel: 'Bostezo promedio',
    avgYawnDesc: 'Duración promedio de todos los bostezos detectados en esta sesión.',
    headPoseLabel: 'Orientación de la cabeza',
    headPoseDesc: 'Inclinación de la cabeza en grados — yaw (izq/der), pitch (arriba/abajo), roll (inclinación).',
    irisDirectionLabel: 'Dirección de la mirada',
    irisDirectionDesc: 'Desplazamiento horizontal y vertical del centro del iris.',
    featuresMissing:
      'Las características biométricas no están presentes en esta evaluación (servidor anterior).',
    earInstantLabel: 'EAR instantáneo',
    earInstantDesc: 'Ratio de apertura ocular en tiempo real, muestreado cuadro a cuadro.',
    currentEyeClosureLabel: 'Cierre actual',
    currentEyeClosureDesc: 'Cuánto tiempo llevan los ojos cerrados continuamente ahora mismo.',
    longestEyeClosureLabel: 'Cierre más largo',
    longestEyeClosureDesc: 'El evento de cierre ocular más largo en esta sesión — valores altos indican riesgo.',
    longestYawnLabel: 'Bostezo más largo',
    longestYawnDesc: 'Duración del bostezo más largo detectado en esta sesión.',
    totalBlinksLabel: 'Parpadeos totales',
    totalBlinksDesc: 'Conteo acumulado de todos los parpadeos desde que inició el pipeline.',
    totalYawnsLabel: 'Bostezos totales',
    totalYawnsDesc: 'Conteo acumulado de todos los bostezos desde que inició el pipeline.',
    // Risk state
    riskStateTitle: 'Estado de riesgo',
    riskStateAria: 'Estado de riesgo en tiempo real',
    riskNormal: 'Normal',
    riskNormalDesc: 'Sin indicadores de fatiga detectados.',
    riskLongBlink: 'Parpadeos largos',
    riskLongBlinkDesc: 'Aparecen cierres oculares de 0,3–0,5 s.',
    riskFatigueWarning: 'Alerta de fatiga',
    riskFatigueWarningDesc: 'Cierre ocular sostenido u otros indicadores de fatiga.',
    riskMicrosleepSuspected: 'Microsueño sospechado',
    riskMicrosleepSuspectedDesc: 'Se detectó un cierre ocular prolongado compatible con microsueño.',
    riskCriticalMicrosleep: 'Microsueño crítico',
    riskCriticalMicrosleepDesc: 'Se detectó un cierre ocular muy prolongado. Actúe de inmediato.',
    // Recent events
    recentEventsTitle: 'Eventos recientes',
    emptyEvents: 'Aún no hay eventos biométricos completados.',
    eventBlink: 'Parpadeo',
    eventLongBlink: 'Parpadeo largo',
    eventEyesClosed: 'Ojos cerrados',
    eventYawn: 'Bostezo',
    // Fatigue
    fatigueTitle: 'Fatiga',
    waitingFatigue: 'Esperando una evaluación de fatiga...',
    fatigueScore: 'Puntaje de fatiga',
    fatigueScoreAria: 'Puntaje de fatiga',
    fatigueLevelAria: 'Nivel de fatiga',
    levelLabel: 'nivel: {l}',
    levelLow: 'bajo',
    levelMedium: 'medio',
    levelHigh: 'alto',
    confidence: 'Confianza',
    // Webcam feed
    noFaceDetected: 'Sin rostro detectado',
    // Overlay controls (S1-18)
    overlayControlsTitle: 'Controles del overlay',
    overlayMirror: 'Vista en espejo (selfie)',
    alertSound: 'Sonido de alertas',
    alertDismiss: 'Cerrar',
    overlayBbox: 'Caja delimitadora',
    overlayLandmarks: 'Puntos faciales (468)',
    overlayEyes: 'Resaltar ojos cerrados',
    overlayMouth: 'Resaltar bostezo',
    overlayGaze: 'Dirección de la mirada',
    overlayHeadPose: 'Gizmo de orientación',
    overlayStatusRing: 'Anillo de estado',
    overlayHud: 'HUD (FPS / latencia / WS)',
    hudFps: 'FPS',
    hudLatency: 'Latencia',
    hudConnection: 'WS',
    // Settings (S4-02)
    settingsTitle: 'Configuración',
    settingsClose: 'Cerrar configuración',
    settingsCameraDevice: 'Dispositivo de cámara',
    settingsDefaultCamera: 'Predeterminada del sistema',
    settingsNoCameras: 'No se detectaron cámaras. Otorga permiso de cámara primero.',
    settingsAlarmVolume: 'Volumen de alarma',
    settingsModelMode: 'Modelo de detección',
    settingsModelModeDesc: 'El modelo ML de fatiga se ejecuta en el servidor. El tipo se configura con la variable de entorno FATIGUE_MODEL_TYPE en el backend.',
    settingsOverlays: 'Capas del overlay',
  },
} satisfies Record<Lang, Record<string, string>>

export type TranslationKey = keyof typeof DICTIONARY['en']

type T = (key: TranslationKey) => string

export interface LanguageContextValue {
  lang: Lang
  t: T
  setLang: (lang: Lang) => void
}

const LanguageContext = createContext<LanguageContextValue | null>(null)

function readStoredLang(): Lang {
  if (typeof window === 'undefined') return 'en'
  const stored = window.localStorage.getItem(LANG_STORAGE_KEY)
  return stored === 'es' ? 'es' : 'en'
}

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(readStoredLang)
  const setLang = useCallback((next: Lang) => {
    setLangState(next)
    if (typeof window !== 'undefined') window.localStorage.setItem(LANG_STORAGE_KEY, next)
  }, [])
  const value = useMemo<LanguageContextValue>(
    () => ({
      lang,
      t: (key: TranslationKey) => DICTIONARY[lang][key],
      setLang,
    }),
    [lang, setLang],
  )
  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>
}

export function useLanguage(): LanguageContextValue {
  const ctx = useContext(LanguageContext)
  if (ctx === null) throw new Error('useLanguage must be used within LanguageProvider')
  return ctx
}

/** Sustituye placeholders {x} de una plantilla traducida con los valores dados. */
export function formatTemplate(
  template: string,
  params: Record<string, string | number>,
): string {
  return template.replace(/\{(\w+)\}/g, (_, name: string) => String(params[name] ?? ''))
}