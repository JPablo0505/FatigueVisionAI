/**
 * SettingsPanel (S4-02, S4-02 fix).
 *
 * Slide-in drawer for user preferences:
 * - Camera device selector
 * - Alarm type selector (beep / siren / emergency / buzzer)
 * - Overlay controls (mirror, sound, all toggles)
 * - Model mode display (informational)
 *
 * All preferences persisted in localStorage. Volume is always 100%.
 */
import { useEffect, useState } from 'react'
import type { OverlayOptions } from './WebcamFeed'
import { useLanguage } from '../i18n'
import type { TranslationKey } from '../i18n'
import { ALARM_TYPES } from '../lib/alertSound'
import type { AlarmType } from '../lib/alertSound'

/* ---------- localStorage keys ---------- */
const CAM_KEY = 'fatiguevision.cameraDevice'
const ALARM_KEY = 'fatiguevision.alarmType'

export const ALERT_VOLUME_KEY = 'fatiguevision.alertVolume'
export const CAMERA_DEVICE_KEY = CAM_KEY

/* ---------- helpers ---------- */
export function readAlertVolume(): number {
  return 100 // Always max — prevent users from silencing alerts
}

export function readCameraDeviceId(): string {
  if (typeof window === 'undefined') return ''
  return window.localStorage.getItem(CAM_KEY) ?? ''
}

export function readAlarmType(): AlarmType {
  if (typeof window === 'undefined') return 'siren'
  const raw = window.localStorage.getItem(ALARM_KEY)
  if (raw === 'beep' || raw === 'siren' || raw === 'emergency' || raw === 'buzzer') return raw
  return 'siren'
}

/* ---------- overlay option labels ---------- */
const OVERLAY_OPTIONS: Array<{ key: keyof OverlayOptions; label: TranslationKey }> = [
  { key: 'bbox', label: 'overlayBbox' },
  { key: 'landmarks', label: 'overlayLandmarks' },
  { key: 'eyes', label: 'overlayEyes' },
  { key: 'mouth', label: 'overlayMouth' },
  { key: 'gaze', label: 'overlayGaze' },
  { key: 'headPose', label: 'overlayHeadPose' },
  { key: 'statusRing', label: 'overlayStatusRing' },
  { key: 'hud', label: 'overlayHud' },
]

/* ---------- props ---------- */
interface SettingsPanelProps {
  open: boolean
  onClose: () => void
  /* alarm type */
  alarmType: AlarmType
  onAlarmTypeChange: (t: AlarmType) => void
  /* overlay options */
  options: OverlayOptions
  onOptionsChange: (o: OverlayOptions) => void
  /* mirror + sound */
  isMirrored: boolean
  onMirroredChange: (v: boolean) => void
  soundEnabled: boolean
  onSoundEnabledChange: (v: boolean) => void
}

/* ---------- component ---------- */
export function SettingsPanel({
  open,
  onClose,
  alarmType,
  onAlarmTypeChange,
  options,
  onOptionsChange,
  isMirrored,
  onMirroredChange,
  soundEnabled,
  onSoundEnabledChange,
}: SettingsPanelProps) {
  const { t } = useLanguage()
  const [devices, setDevices] = useState<Array<{ id: string; label: string }>>([])
  const [selectedDevice, setSelectedDevice] = useState(readCameraDeviceId)

  /* Enumerate camera devices on open */
  useEffect(() => {
    if (!open) return
    let cancelled = false
    async function enumerate() {
      try {
        const media = navigator.mediaDevices
        if (!media?.enumerateDevices) return
        const all = await media.enumerateDevices()
        if (cancelled) return
        const cams = all
          .filter((d) => d.kind === 'videoinput')
          .map((d, i) => ({ id: d.deviceId, label: d.label || `Camera ${i + 1}` }))
        setDevices(cams)
      } catch {
        // Permission not yet granted — leave list empty
      }
    }
    void enumerate()
    return () => { cancelled = true }
  }, [open])

  /* Persist camera device selection */
  const handleDeviceChange = (deviceId: string) => {
    setSelectedDevice(deviceId)
    window.localStorage.setItem(CAM_KEY, deviceId)
  }

  /* Persist alarm type */
  const handleAlarmType = (type: AlarmType) => {
    onAlarmTypeChange(type)
    window.localStorage.setItem(ALARM_KEY, type)
  }

  /* Overlay toggle helper */
  const toggleOverlay = (key: keyof OverlayOptions) => {
    onOptionsChange({ ...options, [key]: !options[key] })
  }

  /* Close on Escape */
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  return (
    <>
      {/* Backdrop */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm transition-opacity"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Drawer */}
      <aside
        role="dialog"
        aria-label={t('settingsTitle')}
        className={`fixed right-0 top-0 z-50 flex h-full w-80 flex-col bg-ebony-clay-900 border-l border-ebony-clay-800 shadow-2xl transition-transform duration-200 ${
          open ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-ebony-clay-800 px-5 py-4">
          <h2 className="text-sm font-semibold text-ebony-clay-100">{t('settingsTitle')}</h2>
          <button
            type="button"
            onClick={onClose}
            aria-label={t('settingsClose')}
            className="rounded-md p-1 text-ebony-clay-400 hover:bg-ebony-clay-800 hover:text-ebony-clay-200"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-5 space-y-6">

          {/* ── Camera device ── */}
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-ebony-clay-400 mb-2">
              {t('settingsCameraDevice')}
            </h3>
            {devices.length === 0 ? (
              <p className="text-xs text-ebony-clay-500">{t('settingsNoCameras')}</p>
            ) : (
              <select
                value={selectedDevice}
                onChange={(e) => handleDeviceChange(e.target.value)}
                className="w-full rounded-lg border border-ebony-clay-700 bg-ebony-clay-800 px-3 py-2 text-sm text-ebony-clay-100 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              >
                <option value="">{t('settingsDefaultCamera')}</option>
                {devices.map((d) => (
                  <option key={d.id} value={d.id}>{d.label}</option>
                ))}
              </select>
            )}
          </section>

          {/* ── Alarm type ── */}
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-ebony-clay-400 mb-2">
              {t('settingsAlarmType')}
            </h3>
            <div className="grid grid-cols-2 gap-2">
              {ALARM_TYPES.map((at) => {
                const active = alarmType === at.id
                return (
                  <button
                    key={at.id}
                    type="button"
                    onClick={() => handleAlarmType(at.id)}
                    className={`rounded-lg border px-3 py-2 text-xs font-medium transition-colors ${
                      active
                        ? 'border-cyan-500 bg-cyan-500/10 text-cyan-400'
                        : 'border-ebony-clay-700 bg-ebony-clay-800 text-ebony-clay-300 hover:border-ebony-clay-600'
                    }`}
                  >
                    {t(at.labelKey as any)}
                  </button>
                )
              })}
            </div>
          </section>

          {/* ── Overlay controls ── */}
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-ebony-clay-400 mb-2">
              {t('overlayControlsTitle')}
            </h3>
            <div className="space-y-2">
              {/* Mirror toggle */}
              <label className="flex items-center justify-between gap-3 text-sm text-ebony-clay-300 cursor-pointer">
                <span>{t('overlayMirror')}</span>
                <input
                  type="checkbox"
                  checked={isMirrored}
                  onChange={(e) => onMirroredChange(e.target.checked)}
                  className="h-4 w-4 accent-cyan-500"
                />
              </label>
              {/* Sound toggle */}
              <label className="flex items-center justify-between gap-3 text-sm text-ebony-clay-300 cursor-pointer">
                <span>{t('alertSound')}</span>
                <input
                  type="checkbox"
                  checked={soundEnabled}
                  onChange={(e) => onSoundEnabledChange(e.target.checked)}
                  className="h-4 w-4 accent-cyan-500"
                />
              </label>
              {/* All overlay options */}
              {OVERLAY_OPTIONS.map(({ key, label }) => (
                <label key={key} className="flex items-center justify-between gap-3 text-sm text-ebony-clay-300 cursor-pointer">
                  <span>{t(label)}</span>
                  <input
                    type="checkbox"
                    checked={options[key]}
                    onChange={() => toggleOverlay(key)}
                    className="h-4 w-4 accent-cyan-500"
                  />
                </label>
              ))}
            </div>
          </section>

          {/* ── Model mode (informational) ── */}
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-ebony-clay-400 mb-2">
              {t('settingsModelMode')}
            </h3>
            <p className="text-xs text-ebony-clay-500">{t('settingsModelModeDesc')}</p>
          </section>
        </div>
      </aside>
    </>
  )
}
