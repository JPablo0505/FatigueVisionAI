/**
 * useRiskCooldown (S4-02 fix, v2).
 *
 * TWO separate outputs:
 * - `displayAssessment` — cooldown-adjusted, used by RiskBanner for visual hold.
 * - `rawAssessment` — the unmodified backend state, used by useFatigueAlert
 *   to detect real escalations. This ensures that when the backend detects
 *   a worse state during cooldown, the alarm still fires.
 *
 * The cooldown ONLY holds the DISPLAY state — it never masks backend escalations
 * from the alert system.
 */
import { useEffect, useRef, useState } from 'react'
import type { FatigueAssessment, RiskState } from '@fatiguevision/contracts'

/** Severity order: higher = more dangerous. */
const SEVERITY: Record<RiskState, number> = {
  normal: 0,
  long_blink: 1,
  fatigue_warning: 2,
  microsleep_suspected: 3,
  critical_microsleep: 4,
}

/**
 * Cooldown durations per severity level (ms).
 * After reaching this state, the DISPLAY holds for this long before allowing
 * a downgrade. The raw state is ALWAYS passed through for alert detection.
 */
const COOLDOWN_MS: Record<RiskState, number> = {
  normal: 0,
  long_blink: 3_000,
  fatigue_warning: 5_000,
  microsleep_suspected: 10_000,
  critical_microsleep: 15_000,
}

export interface RiskCooldownResult {
  /** Cooldown-adjusted assessment for DISPLAY (RiskBanner). */
  displayAssessment: FatigueAssessment | null
  /** Raw assessment from backend for ALERT DETECTION (useFatigueAlert). */
  rawAssessment: FatigueAssessment | null
  /** Whether the display is being held by cooldown. */
  isHeld: boolean
  /** Ms remaining in the current cooldown hold. */
  holdRemainingMs: number
}

export function useRiskCooldown(raw: FatigueAssessment | null): RiskCooldownResult {
  const [heldState, setHeldState] = useState<RiskState | null>(null)
  const [heldSince, setHeldSince] = useState<number>(0)
  const cooldownTimerRef = useRef<number | null>(null)

  const currentRaw = raw?.state ?? 'normal'
  const currentSeverity = SEVERITY[currentRaw]

  useEffect(() => {
    // Clear any pending cooldown timer
    if (cooldownTimerRef.current !== null) {
      window.clearTimeout(cooldownTimerRef.current)
      cooldownTimerRef.current = null
    }

    // If there's no held state OR the new raw state is MORE severe → upgrade
    if (heldState === null || currentSeverity > SEVERITY[heldState]) {
      setHeldState(currentRaw)
      setHeldSince(Date.now())

      const cooldown = COOLDOWN_MS[currentRaw]
      if (cooldown > 0) {
        cooldownTimerRef.current = window.setTimeout(() => {
          cooldownTimerRef.current = null
          setHeldState(null)
        }, cooldown)
      }
      return
    }

    // Still within cooldown window → keep the held state
    if (heldState !== null) {
      const elapsed = Date.now() - heldSince
      const cooldown = COOLDOWN_MS[heldState]
      if (cooldown > 0 && elapsed < cooldown) {
        const remaining = cooldown - elapsed
        cooldownTimerRef.current = window.setTimeout(() => {
          cooldownTimerRef.current = null
          setHeldState(null)
        }, remaining)
        return
      }
    }

    // Cooldown expired — release
    setHeldState(null)
  }, [raw])

  // Display state: cooldown-adjusted (for RiskBanner visual hold)
  const displayState: RiskState = (() => {
    if (heldState !== null) {
      const elapsed = Date.now() - heldSince
      const cooldown = COOLDOWN_MS[heldState]
      if (cooldown > 0 && elapsed < cooldown && SEVERITY[heldState] > currentSeverity) {
        return heldState
      }
    }
    return currentRaw
  })()

  const isHeld = displayState !== currentRaw
  const holdRemainingMs = isHeld && heldState !== null
    ? Math.max(0, COOLDOWN_MS[heldState] - (Date.now() - heldSince))
    : 0

  // Display assessment: cooldown-adjusted (RiskBanner sees this)
  const displayAssessment: FatigueAssessment | null = raw === null
    ? null
    : displayState === raw.state
      ? raw
      : { ...raw, state: displayState }

  // Raw assessment: ALWAYS the real backend state (useFatigueAlert sees this)
  // This ensures escalations are never masked by cooldown.
  return { displayAssessment, rawAssessment: raw, isHeld, holdRemainingMs }
}
