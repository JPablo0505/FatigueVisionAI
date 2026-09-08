import { useEffect, useMemo, useState } from 'react'
import { Alert, Badge, Button, Loading, Stack, StatusIndicator } from '@fatiguevision/ui'
import { PROTOCOL_VERSION } from '@fatiguevision/contracts'
import type {
  BiometricEvent,
  FatigueAssessment,
  FaceOverlay,
  FatigueLevel,
  RiskState,
  VisionStatus,
} from '@fatiguevision/contracts'
import { buildControlStart, buildControlStop, buildPing } from './lib/envelope'
import { useFatigueStream } from './hooks/useFatigueStream'
import type { ConnectionState } from './hooks/useFatigueStream'
import { useCameraStream } from './hooks/useCameraStream'
import { useOverlayFps } from './hooks/useOverlayFps'
import { WebcamFeed } from './components/WebcamFeed'
import type { OverlayConnection, OverlayOptions } from './components/WebcamFeed'
import { OverlayControls } from './components/OverlayControls'
import { FatigueAlert } from './components/FatigueAlert'
import { FatigueGauge } from './components/FatigueGauge'
import { Panel } from './components/Panel'
import { MetricRow } from './components/MetricRow'
import { MetricCell } from './components/MetricCell'
import { FatigueLogo } from './components/FatigueLogo'
import { SettingsPanel, readAlertVolume, readAlarmType } from './components/SettingsPanel'
import { alertSound } from './lib/alertSound'
import type { AlarmType } from './lib/alertSound'
import { useRiskCooldown } from './hooks/useRiskCooldown'
import { formatTemplate, useLanguage } from './i18n'
import type { Lang, TranslationKey } from './i18n'
import { useFatigueAlert } from './hooks/useFatigueAlert'

type DotStatus = 'idle' | 'ok' | 'warn' | 'error'
type BadgeTone = 'neutral' | 'info' | 'success' | 'warning' | 'error'
type T = (key: TranslationKey) => string

const LEVEL_TONE: Record<FatigueLevel, BadgeTone> = {
  low: 'success',
  medium: 'warning',
  high: 'error',
}

/** Color de fondo del RiskBanner por estado de riesgo (S1-17). */
const RISK_COLOR: Record<RiskState, string> = {
  normal: 'bg-emerald-500',
  long_blink: 'bg-amber-500',
  fatigue_warning: 'bg-orange-500',
  microsleep_suspected: 'bg-red-500',
  critical_microsleep: 'bg-red-600',
}

/** Color del dot por tipo de evento biométrico (S1-17). */
const EVENT_COLOR: Record<BiometricEvent['type'], string> = {
  blink: 'bg-slate-500',
  long_blink: 'bg-amber-600',
  eyes_closed: 'bg-red-500',
  yawn: 'bg-cyan-600',
}

/** Toggles del overlay (S1-18): estado local del frontend, SOLO visual. */
const DEFAULT_OVERLAY_OPTIONS: OverlayOptions = {
  bbox: true,
  landmarks: true,
  eyes: true,
  mouth: true,
  gaze: true,
  headPose: true,
  statusRing: true,
  hud: true,
}

/** Clave de localStorage de la preferencia de sonido de alertas (S1-19). */
const SOUND_STORAGE_KEY = 'fatiguevision.alertSound'

/** Preferencia de sonido persistida; por defecto activado. */
function readSoundEnabled(): boolean {
  if (typeof window === 'undefined') return true
  return window.localStorage.getItem(SOUND_STORAGE_KEY) !== '0'
}

function riskName(t: T, state: RiskState): string {
  switch (state) {
    case 'normal':
      return t('riskNormal')
    case 'long_blink':
      return t('riskLongBlink')
    case 'fatigue_warning':
      return t('riskFatigueWarning')
    case 'microsleep_suspected':
      return t('riskMicrosleepSuspected')
    case 'critical_microsleep':
      return t('riskCriticalMicrosleep')
  }
}

function riskDescription(t: T, state: RiskState): string {
  switch (state) {
    case 'normal':
      return t('riskNormalDesc')
    case 'long_blink':
      return t('riskLongBlinkDesc')
    case 'fatigue_warning':
      return t('riskFatigueWarningDesc')
    case 'microsleep_suspected':
      return t('riskMicrosleepSuspectedDesc')
    case 'critical_microsleep':
      return t('riskCriticalMicrosleepDesc')
  }
}

const EVENT_NAME: Record<BiometricEvent['type'], (t: T) => string> = {
  blink: (t) => t('eventBlink'),
  long_blink: (t) => t('eventLongBlink'),
  eyes_closed: (t) => t('eventEyesClosed'),
  yawn: (t) => t('eventYawn'),
}

function eventName(t: T, type: BiometricEvent['type']): string {
  return EVENT_NAME[type]!(t)
}

function connectionStatus(t: T): Record<ConnectionState, { dot: DotStatus; label: string }> {
  return {
    connecting: { dot: 'warn', label: t('connecting') },
    open: { dot: 'ok', label: t('connected') },
    closing: { dot: 'warn', label: t('closing') },
    closed: { dot: 'error', label: t('disconnected') },
  }
}

function pipelineStatus(t: T): Record<VisionStatus['pipeline'], { dot: DotStatus; label: string }> {
  return {
    running: { dot: 'ok', label: t('running') },
    stopped: { dot: 'idle', label: t('stopped') },
    error: { dot: 'error', label: t('errorState') },
  }
}

function levelName(t: T, level: FatigueLevel): string {
  switch (level) {
    case 'low':
      return t('levelLow')
    case 'medium':
      return t('levelMedium')
    case 'high':
      return t('levelHigh')
  }
}

/** Formatea un número con N decimales (solo display; no calcula nada). */
function fmt(value: number, digits: number): string {
  return value.toFixed(digits)
}

function formatEpochSeconds(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleTimeString()
}

function LanguageSelector() {
  const { t, lang, setLang } = useLanguage()
  const options: Lang[] = ['en', 'es']
  return (
    <div
      role="group"
      aria-label={t('languageSelector')}
      className="flex rounded-lg border border-ebony-clay-800 bg-ebony-clay-900 p-0.5"
    >
      {options.map((code) => {
        const active = lang === code
        return (
          <button
            key={code}
            type="button"
            aria-pressed={active}
            onClick={() => setLang(code)}
            className={
              active
                ? 'rounded-md bg-ebony-clay-700 px-3 py-1 text-xs font-semibold text-ebony-clay-100'
                : 'rounded-md px-3 py-1 text-xs font-medium text-ebony-clay-400 hover:text-ebony-clay-200'
            }
          >
            {code.toUpperCase()}
          </button>
        )
      })}
    </div>
  )
}

function App() {
  const { t } = useLanguage()
  const { connectionState, reconnectAttempt, visionStatus, assessment, overlay, serverError, lastPongAt, events, send } =
    useFatigueStream()
  const { stream, error: cameraError } = useCameraStream()
  const overlayFps = useOverlayFps(overlay)
  const [overlayOptions, setOverlayOptions] = useState<OverlayOptions>(DEFAULT_OVERLAY_OPTIONS)
  // Vista en espejo (S1-18 UX): el dashboard se percibe como selfie, así que
  // por defecto la flecha de mirada y el yaw del gizmo se invierten en X para
  // seguir la dirección percibida. Toggle de presentación pura (false = datos
  // reales del backend, sin inversión).
  const [isMirrored, setIsMirrored] = useState(true)
  // Sonido de alertas temporales (S1-19): preferencia persistida, se sincroniza
  // con el motor de Web Audio y el toggle de OverlayControls.
  const [soundEnabled, setSoundEnabled] = useState<boolean>(readSoundEnabled)

  // S4-02 fix: Risk cooldown — prevents instant recovery after microsleep
  const { displayAssessment, rawAssessment, isHeld: isRiskHeld, holdRemainingMs } = useRiskCooldown(assessment)

  const { alert, dismiss } = useFatigueAlert(rawAssessment, soundEnabled)

  // S4-02: Settings panel & alarm config
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [volume, setVolume] = useState(readAlertVolume)
  const [alarmType, setAlarmType] = useState<AlarmType>(readAlarmType)

  // S4-01: Flash visual rojo cuando el estado es critical_microsleep
  const isCritical = assessment?.state === 'critical_microsleep'

  useEffect(() => {
    window.localStorage.setItem(SOUND_STORAGE_KEY, soundEnabled ? '1' : '0')
  }, [soundEnabled])

  // S4-02: Sync volume to the sound engine
  useEffect(() => {
    alertSound.setVolume(volume)
  }, [volume])

  // S4-02: Sync alarm type to the sound engine
  useEffect(() => {
    alertSound.setType(alarmType)
  }, [alarmType])

  const connected = connectionState === 'open'
  const isRunning = visionStatus?.pipeline === 'running'
  const browserMode = visionStatus?.camera.source === 'browser'

  // En modo browser, el backend no abre cámara: el navegador sube cada frame
  // mientras el pipeline esté corriendo y la conexión siga abierta.
  const frameProvide = useMemo(
    () => (connected && isRunning && browserMode ? { active: true, send } : null),
    [connected, isRunning, browserMode, send],
  )

  // Datos de presentación para el overlay (S1-18): FPS medidos, latencia
  // now - timestamp del overlay (igual que SystemStatusPanel) y estado de la
  // conexión como tri-estado simple para el HUD.
  const overlayLatencyMs = overlay === null ? 0 : Date.now() - overlay.timestamp_ns / 1e6
  const overlayConnection: OverlayConnection =
    connectionState === 'open' ? 'connected' : connectionState === 'connecting' ? 'connecting' : 'disconnected'
  // assessment.state llega ya tipado del hook; acá solo se mapea a {code, label}.
  const risk = assessment?.state
  const overlayState = risk === undefined ? null : { code: risk, label: riskName(t, risk) }

  const connection =
    connectionState === 'closed' && reconnectAttempt > 0
      ? {
          dot: 'warn' as const,
          label: `${t('reconnecting')} (${formatTemplate(t('attemptCount'), { n: reconnectAttempt })})`,
        }
      : connectionStatus(t)[connectionState]

  const pipeline = visionStatus === null ? null : pipelineStatus(t)[visionStatus.pipeline]
  const pipelineLabel = pipeline === null ? t('na') : pipeline.label

  const handleStart = () => {
    send(buildControlStart())
  }

  const handleStop = () => {
    send(buildControlStop())
  }

  const handlePing = () => {
    send(buildPing())
  }

  return (
    <div className="min-h-dvh bg-ebony-clay-950 p-6 text-slate-100">
      {/* S4-01: Critical alarm — red screen flash overlay */}
      {isCritical && (
        <div
          className="pointer-events-none fixed inset-0 z-[100] bg-red-600 animate-critical-flash"
          aria-hidden="true"
        />
      )}
      <FatigueAlert alert={alert} onDismiss={dismiss} />
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6">
        <header className="flex flex-wrap items-center justify-between gap-4 rounded-2xl bg-ebony-clay-900 px-6 py-4">
          <div className="flex items-center gap-3">
            <FatigueLogo className="h-9 w-9" />
            <div>
              <h1 className="text-2xl font-semibold text-ebony-clay-50">FatigueVision AI</h1>
              <p className="mt-0.5 text-sm text-ebony-clay-400">{t('subtitle')}</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <LanguageSelector />
            <button
              type="button"
              onClick={() => setSettingsOpen(true)}
              aria-label={t('settingsTitle')}
              className="rounded-lg border border-ebony-clay-800 bg-ebony-clay-900 p-2 text-ebony-clay-400 hover:bg-ebony-clay-800 hover:text-ebony-clay-200"
            >
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </button>
            <Badge tone="info" ariaLabel={t('protocolBadge')}>
              {formatTemplate(t('protocolText'), { v: PROTOCOL_VERSION })}
            </Badge>
            <StatusIndicator status={connection.dot} label={connection.label} className="[&_span]:text-ebony-clay-200" />
            <StatusIndicator
              status={pipeline === null ? 'idle' : pipeline.dot}
              label={`${t('pipeline')}: ${pipelineLabel}`}
              className="[&_span]:text-ebony-clay-200"
            />
            <Badge tone="neutral" ariaLabel={t('lastUpdate')}>
              {lastPongAt !== null
                ? formatTemplate(t('lastUpdate'), { t: formatEpochSeconds(lastPongAt) })
                : t('lastUpdateNone')}
            </Badge>
          </div>
        </header>

        {connectionState === 'closed' ? (
          <Alert tone="warning" title={t('connectionErrorTitle')}>
            {t('connectionErrorBody')}
          </Alert>
        ) : null}

        {serverError !== null ? (
          <Alert tone="error" title={formatTemplate(t('serverErrorTitle'), { code: serverError.code })}>
            {serverError.message}
          </Alert>
        ) : null}

        {visionStatus?.pipeline === 'error' ? (
          <Alert tone="error" title={t('pipelineErrorTitle')}>
            {visionStatus.error ?? t('unknownPipelineError')}
          </Alert>
        ) : null}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
          <div className="flex flex-col gap-6">
            {stream === null ? (
              <CameraPlaceholder error={cameraError} />
            ) : (
              <div className="rounded-2xl border border-ebony-clay-800 bg-ebony-clay-900 p-3">
                <WebcamFeed
                  stream={stream}
                  overlay={overlay}
                  fps={overlayFps}
                  latencyMs={overlayLatencyMs}
                  connection={overlayConnection}
                  options={overlayOptions}
                  state={overlayState}
                  isMirrored={isMirrored}
                  frameProvide={frameProvide}
                  isStreaming={isRunning}
                />
              </div>
            )}

            <OverlayControls
              options={overlayOptions}
              onChange={setOverlayOptions}
              isMirrored={isMirrored}
              onMirroredChange={setIsMirrored}
              soundEnabled={soundEnabled}
              onSoundEnabledChange={setSoundEnabled}
            />

            <Panel title={t('controlsTitle')}>
              <Stack direction="row" gap={4}>
                <Button variant="primary" onClick={handleStart} disabled={!connected || isRunning}>
                  {t('start')}
                </Button>
                <Button variant="danger" onClick={handleStop} disabled={!connected || !isRunning}>
                  {t('stop')}
                </Button>
                <Button variant="secondary" onClick={handlePing} disabled={!connected}>
                  {t('ping')}
                </Button>
              </Stack>
              <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-ebony-clay-800 pt-4 text-sm text-ebony-clay-400">
                <StatusIndicator
                  status={pipeline === null ? 'idle' : pipeline.dot}
                  label={`${t('pipeline')}: ${pipelineLabel}`}
                  className="[&_span]:text-ebony-clay-300"
                />
                <span>
                  {t('pingKeepAlive')}
                  {lastPongAt !== null ? formatTemplate(t('lastPong'), { t: formatEpochSeconds(lastPongAt) }) : ''}
                </span>
              </div>
            </Panel>
          </div>

          <div className="flex flex-col gap-6">
            <RiskBanner assessment={displayAssessment} isHeld={isRiskHeld} holdRemainingMs={holdRemainingMs} />
            <SystemStatusPanel visionStatus={visionStatus} overlay={overlay} fps={overlayFps} />
            <BiometricsPanel assessment={assessment} />
            <FatiguePanel assessment={assessment} />
            <RecentEventsPanel events={events} />
          </div>
        </div>
      </div>

      {/* S4-02: Settings drawer */}
      <SettingsPanel
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        volume={volume}
        onVolumeChange={setVolume}
        alarmType={alarmType}
        onAlarmTypeChange={setAlarmType}
        options={overlayOptions}
        onOptionsChange={setOverlayOptions}
      />
    </div>
  )
}

interface CameraPlaceholderProps {
  error: string | null
}

function CameraPlaceholder({ error }: CameraPlaceholderProps) {
  const { t } = useLanguage()
  const message = error ?? t('cameraPlaceholderHint')
  return (
    <div className="flex aspect-[4/3] flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-ebony-clay-700 bg-ebony-clay-900 p-6 text-center">
      <span className="text-sm font-semibold text-ebony-clay-200">{t('cameraUnavailable')}</span>
      <p className="max-w-sm text-sm text-ebony-clay-400">{message}</p>
    </div>
  )
}

interface SystemStatusPanelProps {
  visionStatus: VisionStatus | null
  overlay: FaceOverlay | null
  fps: number
}

function SystemStatusPanel({ visionStatus, overlay, fps }: SystemStatusPanelProps) {
  const { t } = useLanguage()
  const pipeline = visionStatus === null ? null : pipelineStatus(t)[visionStatus.pipeline]
  const latencyMs = overlay === null ? null : Date.now() - overlay.timestamp_ns / 1e6

  return (
    <Panel
      title={t('systemStatusTitle')}
      right={
        pipeline === null ? null : (
          <StatusIndicator status={pipeline.dot} label={pipeline.label} className="[&_span]:text-ebony-clay-200" />
        )
      }
    >
      {visionStatus === null ? (
        <Loading label={t('waitingPipelineStatus')} className="[&_span]:text-ebony-clay-300" />
      ) : (
        <dl className="flex flex-col gap-3">
          <MetricRow
            label={t('faceDetected')}
            value={overlay === null ? t('no') : formatTemplate(t('facesDetected'), { n: overlay.face_count })}
          />
          <MetricRow label={t('streamFps')} value={`${fps} fps`} />
          <MetricRow label={t('latency')} value={latencyMs === null ? '—' : `${fmt(latencyMs, 1)} ms`} />
          <MetricRow
            label={t('camera')}
            value={`${visionStatus.camera.source} · ${visionStatus.camera.running ? t('runningVerb') : t('idleVerb')}`}
          />
        </dl>
      )}
    </Panel>
  )
}

interface RiskBannerProps {
  assessment: FatigueAssessment | null
  isHeld?: boolean
  holdRemainingMs?: number
}

function RiskBanner({ assessment, isHeld, holdRemainingMs }: RiskBannerProps) {
  const { t } = useLanguage()
  if (assessment === null) {
    return (
      <div
        role="status"
        aria-label={t('riskStateAria')}
        className="rounded-2xl border border-ebony-clay-800 bg-ebony-clay-900 p-4"
      >
        <span className="text-xs font-bold uppercase tracking-wide text-ebony-clay-400">{t('riskStateTitle')}</span>
        <p className="mt-2 text-sm text-ebony-clay-400">{t('waitingAssessment')}</p>
      </div>
    )
  }
  const state = assessment.state ?? 'normal'
  const holdSeconds = holdRemainingMs !== undefined && holdRemainingMs > 0
    ? Math.ceil(holdRemainingMs / 1000)
    : 0
  return (
    <div
      role="status"
      aria-label={t('riskStateAria')}
      className={`rounded-2xl border border-ebony-clay-800 p-4 text-slate-950 ${RISK_COLOR[state]}`}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-xs font-bold uppercase tracking-wide">{t('riskStateTitle')}</span>
        <div className="flex items-center gap-2">
          {isHeld && holdSeconds > 0 && (
            <span className="rounded bg-black/20 px-1.5 py-0.5 text-[10px] font-bold tabular-nums">
              HOLD {holdSeconds}s
            </span>
          )}
          <span className="text-sm font-bold">{riskName(t, state)}</span>
        </div>
      </div>
      <p className="mt-2 text-sm font-medium text-slate-900">{riskDescription(t, state)}</p>
    </div>
  )
}

interface RecentEventsPanelProps {
  events: BiometricEvent[]
}

function RecentEventsPanel({ events }: RecentEventsPanelProps) {
  const { t } = useLanguage()
  return (
    <Panel title={t('recentEventsTitle')}>
      {events.length === 0 ? (
        <p className="text-sm text-ebony-clay-500">{t('emptyEvents')}</p>
      ) : (
        <ul className="flex flex-col gap-2.5">
          {[...events]
            .reverse()
            .slice(0, 10)
            .map((event, index) => (
              <li key={index} className="flex items-center justify-between gap-2 text-sm">
                <span className="flex items-center gap-2">
                  <span aria-hidden className={`h-2 w-2 shrink-0 rounded-full ${EVENT_COLOR[event.type]}`} />
                  {eventName(t, event.type)}
                </span>
                <span className="tabular-nums text-ebony-clay-400">
                  {fmt(event.duration_seconds, 2)}s · {formatEpochSeconds(event.timestamp)}
                </span>
              </li>
            ))}
        </ul>
      )}
    </Panel>
  )
}

interface BiometricsPanelProps {
  assessment: FatigueAssessment | null
}

function BiometricsPanel({ assessment }: BiometricsPanelProps) {
  const { t } = useLanguage()
  const features = assessment?.features
  const desc = (key: TranslationKey) => t(key)
  return (
    <Panel title={t('biometricsTitle')}>
      {assessment === null ? (
        <Loading label={t('waitingAssessment')} className="[&_span]:text-ebony-clay-300" />
      ) : (
        <>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-5">
            <MetricCell
              label={t('earLabel')}
              value={features === undefined ? t('na') : fmt(features.ear, 3)}
              description={desc('earDesc')}
            />
            <MetricCell
              label={t('earInstantLabel')}
              value={features === undefined ? t('na') : fmt(features.ear_instant ?? 0, 3)}
              description={desc('earInstantDesc')}
            />
            <MetricCell
              label={t('marLabel')}
              value={
                features === undefined
                  ? t('na')
                  : `${fmt(features.mar_instant ?? 0, 3)} · ${t('avg')} ${fmt(features.mar, 3)}`
              }
              description={desc('marDesc')}
            />
            <MetricCell
              label={t('perclosLabel')}
              value={features === undefined ? t('na') : fmt(features.perclos, 3)}
              description={desc('perclosDesc')}
            />
            <MetricCell
              label={t('blinkRateLabel')}
              value={features === undefined ? t('na') : `${fmt(features.blink_rate, 2)}/min`}
              description={desc('blinkRateDesc')}
            />
            <MetricCell
              label={t('eyeClosureLabel')}
              value={features === undefined ? t('na') : `${fmt(features.eye_closure_duration, 2)}s`}
              description={desc('eyeClosureDesc')}
            />
            <MetricCell
              label={t('currentEyeClosureLabel')}
              value={
                features === undefined
                  ? t('na')
                  : `${fmt(features.current_eye_closure_duration ?? 0, 2)}s`
              }
              description={desc('currentEyeClosureDesc')}
            />
            <MetricCell
              label={t('longestEyeClosureLabel')}
              value={
                features === undefined
                  ? t('na')
                  : `${fmt(features.longest_eye_closure_duration ?? 0, 2)}s`
              }
              description={desc('longestEyeClosureDesc')}
            />
            <MetricCell
              label={t('currentYawnLabel')}
              value={features === undefined ? t('na') : `${fmt(features.current_yawn_duration ?? 0, 2)}s`}
              description={desc('currentYawnDesc')}
            />
            <MetricCell
              label={t('lastYawnLabel')}
              value={features === undefined ? t('na') : `${fmt(features.last_yawn_duration ?? 0, 2)}s`}
              description={desc('lastYawnDesc')}
            />
            <MetricCell
              label={t('avgYawnLabel')}
              value={features === undefined ? t('na') : `${fmt(features.yawn_duration, 2)}s`}
              description={desc('avgYawnDesc')}
            />
            <MetricCell
              label={t('longestYawnLabel')}
              value={features === undefined ? t('na') : `${fmt(features.longest_yawn_duration ?? 0, 2)}s`}
              description={desc('longestYawnDesc')}
            />
            <MetricCell
              label={t('totalBlinksLabel')}
              value={features === undefined ? t('na') : fmt(features.total_blinks ?? 0, 0)}
              description={desc('totalBlinksDesc')}
            />
            <MetricCell
              label={t('totalYawnsLabel')}
              value={features === undefined ? t('na') : fmt(features.total_yawns ?? 0, 0)}
              description={desc('totalYawnsDesc')}
            />
          </dl>
          {features === undefined ? (
            <p className="mt-4 text-xs text-ebony-clay-500">{t('featuresMissing')}</p>
          ) : null}
          <div className="mt-4 border-t border-ebony-clay-800 pt-3">
            <dl className="flex flex-col gap-2 text-xs text-ebony-clay-400">
              <div className="flex items-center justify-between gap-2">
                <dt>{t('headPoseLabel')}</dt>
                <dd className="tabular-nums text-ebony-clay-300">
                  {features === undefined
                    ? t('na')
                    : `${fmt(features.head_pose.yaw, 1)}° ${fmt(features.head_pose.pitch, 1)}° ${fmt(features.head_pose.roll, 1)}°`}
                </dd>
              </div>
              <div className="flex items-center justify-between gap-2">
                <dt>{t('irisDirectionLabel')}</dt>
                <dd className="tabular-nums text-ebony-clay-300">
                  {features === undefined
                    ? t('na')
                    : `x ${fmt(features.iris_direction.x, 2)} · y ${fmt(features.iris_direction.y, 2)}`}
                </dd>
              </div>
            </dl>
          </div>
        </>
      )}
    </Panel>
  )
}

interface FatiguePanelProps {
  assessment: FatigueAssessment | null
}

function FatiguePanel({ assessment }: FatiguePanelProps) {
  const { t } = useLanguage()
  return (
    <Panel title={t('fatigueTitle')}>
      {assessment === null ? (
        <Loading label={t('waitingFatigue')} className="[&_span]:text-ebony-clay-300" />
      ) : (
        <>
          <FatigueGauge score={assessment.score} level={assessment.level} />
          <div className="mt-5 flex items-center justify-between border-t border-ebony-clay-800 pt-4">
            <Badge tone={LEVEL_TONE[assessment.level]} ariaLabel={t('fatigueLevelAria')}>
              {formatTemplate(t('levelLabel'), { l: levelName(t, assessment.level) })}
            </Badge>
            <MetricRow label={t('confidence')} value={fmt(assessment.confidence, 3)} />
          </div>
        </>
      )}
    </Panel>
  )
}

export default App