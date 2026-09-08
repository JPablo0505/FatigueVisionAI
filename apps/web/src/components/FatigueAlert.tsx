import { useEffect, useRef, useState } from 'react'
import type { RiskState } from '@fatiguevision/contracts'
import { useLanguage } from '../i18n'
import type { TranslationKey } from '../i18n'
import type { FatigueAlertInfo } from '../hooks/useFatigueAlert'

interface FatigueAlertProps {
  alert: FatigueAlertInfo | null
  onDismiss: () => void
}

/**
 * Color/emoji por estado de riesgo. Réplica LOCAL de STATUS_STYLE
 * (WebcamFeed.tsx:89) para no acoplar el toast al canvas. Mantener en sync.
 */
const ALERT_STYLE: Record<RiskState, { color: string; emoji: string }> = {
  normal: { color: '#10b981', emoji: '🟢' },
  long_blink: { color: '#f59e0b', emoji: '🟡' },
  fatigue_warning: { color: '#f97316', emoji: '🟠' },
  microsleep_suspected: { color: '#ef4444', emoji: '🔴' },
  critical_microsleep: { color: '#b91c1c', emoji: '🚨' },
}

/** Duración de la transición de opacidad antes de desmontar el toast. */
const FADE_OUT_MS = 400

type T = (key: TranslationKey) => string

/** Etiqueta del estado (mismas claves i18n que RiskBanner). */
function stateName(state: RiskState, t: T): string {
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

/** Descripción del estado (mismas claves i18n que RiskBanner). */
function stateDescription(state: RiskState, t: T): string {
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

/**
 * Toast de alerta temporal de fatiga (S1-19). Flotante (fixed), no ocupa
 * layout, se desvanece con transiciones CSS y permite cierre manual. La
 * alerta crítica en curso puede reaparecer vía el hook mientras dure el
 * microsueño.
 */
export function FatigueAlert({ alert, onDismiss }: FatigueAlertProps) {
  const { t } = useLanguage()
  const [display, setDisplay] = useState<FatigueAlertInfo | null>(alert)
  const [visible, setVisible] = useState(alert !== null)
  const unmountTimerRef = useRef<number | null>(null)
  const rafRef = useRef<number | null>(null)

  useEffect(() => {
    if (unmountTimerRef.current !== null) {
      window.clearTimeout(unmountTimerRef.current)
      unmountTimerRef.current = null
    }
    if (rafRef.current !== null) {
      window.cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    if (alert !== null) {
      setDisplay(alert)
      setVisible(false)
      // Doble rAF: dejar que el navegador pinte con opacity-0 y recién entonces
      // transicionar a opacity-100 (fade-in suave, incluye al reemplazar).
      rafRef.current = window.requestAnimationFrame(() => {
        rafRef.current = window.requestAnimationFrame(() => {
          rafRef.current = null
          setVisible(true)
        })
      })
    } else {
      setVisible(false)
      unmountTimerRef.current = window.setTimeout(() => {
        unmountTimerRef.current = null
        setDisplay(null)
      }, FADE_OUT_MS)
    }
  }, [alert])

  useEffect(
    () => () => {
      if (unmountTimerRef.current !== null) window.clearTimeout(unmountTimerRef.current)
      if (rafRef.current !== null) window.cancelAnimationFrame(rafRef.current)
    },
    [],
  )

  if (display === null) return null

  const style = ALERT_STYLE[display.state]
  return (
    <div
      role="alert"
      className={`pointer-events-none fixed right-4 top-4 z-50 w-96 max-w-[calc(100vw-2rem)] rounded-2xl border-2 border-ebony-clay-900/20 p-4 text-ebony-clay-950 shadow-2xl transition-all duration-[400ms] ease-in-out ${
        visible ? 'translate-y-0 opacity-100' : 'translate-y-2 opacity-0'
      }`}
      style={{ backgroundColor: style.color }}
    >
      <div className="flex items-start gap-3">
        <span aria-hidden className="text-2xl leading-none">
          {style.emoji}
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-bold uppercase tracking-wide">{stateName(display.state, t)}</p>
          <p className="mt-1 text-sm font-medium">{stateDescription(display.state, t)}</p>
        </div>
        <button
          type="button"
          onClick={onDismiss}
          aria-label={t('alertDismiss')}
          className="pointer-events-auto -m-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-ebony-clay-900/70 transition-colors hover:bg-ebony-clay-900/10 hover:text-ebony-clay-950"
        >
          <span aria-hidden>✕</span>
        </button>
      </div>
    </div>
  )
}