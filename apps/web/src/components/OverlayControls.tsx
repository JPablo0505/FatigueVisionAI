import { Stack } from '@fatiguevision/ui'
import type { OverlayOptions } from './WebcamFeed'
import { Panel } from './Panel'
import { useLanguage } from '../i18n'
import type { TranslationKey } from '../i18n'

interface OverlayControlsProps {
  options: OverlayOptions
  onChange: (next: OverlayOptions) => void
  /** Vista en espejo (selfie): invierte la flecha de mirada y el yaw en X. */
  isMirrored: boolean
  onMirroredChange: (value: boolean) => void
  /** Sonido de alertas temporales (Web Audio API, sin archivos). */
  soundEnabled: boolean
  onSoundEnabledChange: (value: boolean) => void
}

const OPTION_LABELS: Array<{ key: keyof OverlayOptions; label: TranslationKey }> = [
  { key: 'bbox', label: 'overlayBbox' },
  { key: 'landmarks', label: 'overlayLandmarks' },
  { key: 'eyes', label: 'overlayEyes' },
  { key: 'mouth', label: 'overlayMouth' },
  { key: 'gaze', label: 'overlayGaze' },
  { key: 'headPose', label: 'overlayHeadPose' },
  { key: 'statusRing', label: 'overlayStatusRing' },
  { key: 'hud', label: 'overlayHud' },
]

/**
 * Panel de toggles del overlay (S1-18). Son switches del frontend SOLO
 * visuales: cambian qué capas dibuja el canvas, nunca lo que procesa o
 * emite el backend. El toggle de espejo (S1-18 UX) también es presentación
 * pura: invierte la flecha de mirada y el yaw en X, sin tocar los datos.
 */
export function OverlayControls({
  options,
  onChange,
  isMirrored,
  onMirroredChange,
  soundEnabled,
  onSoundEnabledChange,
}: OverlayControlsProps) {
  const { t } = useLanguage()

  const toggle = (key: keyof OverlayOptions, value: boolean) => {
    onChange({ ...options, [key]: value })
  }

  return (
    <Panel title={t('overlayControlsTitle')}>
      <Stack direction="column" gap={2}>
        <label className="flex cursor-pointer items-center justify-between gap-3 text-sm text-slate-300">
          <span>{t('overlayMirror')}</span>
          <input
            type="checkbox"
            checked={isMirrored}
            onChange={(event) => onMirroredChange(event.target.checked)}
            className="h-4 w-4 accent-cyan-500"
          />
        </label>
        <label className="flex cursor-pointer items-center justify-between gap-3 text-sm text-slate-300">
          <span>{t('alertSound')}</span>
          <input
            type="checkbox"
            checked={soundEnabled}
            onChange={(event) => onSoundEnabledChange(event.target.checked)}
            className="h-4 w-4 accent-cyan-500"
          />
        </label>
        {OPTION_LABELS.map(({ key, label }) => (
          <label
            key={key}
            className="flex cursor-pointer items-center justify-between gap-3 text-sm text-slate-300"
          >
            <span>{t(label)}</span>
            <input
              type="checkbox"
              checked={options[key]}
              onChange={(event) => toggle(key, event.target.checked)}
              className="h-4 w-4 accent-cyan-500"
            />
          </label>
        ))}
      </Stack>
    </Panel>
  )
}