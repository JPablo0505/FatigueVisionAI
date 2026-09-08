import type { FatigueLevel } from '@fatiguevision/contracts'
import { useLanguage } from '../i18n'

interface FatigueGaugeProps {
  score: number
  level: FatigueLevel
}

const LEVEL_FILL: Record<FatigueLevel, string> = {
  low: 'bg-green-500',
  medium: 'bg-amber-500',
  high: 'bg-red-500',
}

/** Barra de progreso horizontal del Fatigue Score [0..100], color según nivel. */
export function FatigueGauge({ score, level }: FatigueGaugeProps) {
  const { t } = useLanguage()
  const clamped = Math.min(100, Math.max(0, score))
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-xs uppercase tracking-wide text-ebony-clay-400">{t('fatigueScore')}</span>
        <span className="text-sm font-semibold tabular-nums text-ebony-clay-100">{score.toFixed(1)} / 100</span>
      </div>
      <div className="h-3 w-full overflow-hidden rounded-full bg-ebony-clay-800">
        <div
          role="progressbar"
          aria-label={t('fatigueScoreAria')}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={clamped}
          className={`h-full rounded-full transition-[width] duration-200 ease-out ${LEVEL_FILL[level]}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  )
}