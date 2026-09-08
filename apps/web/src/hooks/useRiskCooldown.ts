/**
 * useRiskCooldown (S4-02 fix).
 *
 * Wraps the raw assessment from the backend and enforces a minimum hold time
 * for elevated risk states. When the user opens their eyes after a microsleep,
 * the risk state stays elevated for COOLDOWN_MS before allowing a downgrade.
 *
 * This prevents the "instant green" problem where the dashboard immediately
 * shows normal after a critical microsleep ends.
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
 * After reaching this state, it cannot drop below the threshold for this long.
 * - critical_microsleep: 15s cooldown (longest — most dangerous)
 * - microsleep_suspected: 10s
 * - fatigue_warning: 5s
 * - long_blink: 3s
 * - normal: 0 (no cooldown)
 */
const COOLDOWN_MS: Record<RiskState, number> = {
  normal: 0,
  long_blink: 3_000,
  fatigue_warning: 5_000,
  microsleep_suspected: 10_000,
  critical_microsleep: 15_000,
}

export interface RiskCooldownResult {
  /** The cooldown-adjusted assessment (state may be held higher than raw). */
  assessment: FatigueAssessment | null
  /** Whether the currently displayed state is being held by cooldown. */
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

    // If the new raw state is MORE severe than what we're holding, upgrade immediately
    if (heldState === null || currentSeverity > SEVERITY[heldState]) {
      setHeldState(currentRaw)
      setHeldSince(Date.now())

      // Set a timer to re-evaluate after the cooldown for this level
      const cooldown = COOLDOWN_MS[currentRaw]
      if (cooldown > 0) {
        cooldownTimerRef.current = window.setTimeout(() => {
          cooldownTimerRef.current = null
          setHeldState(null) // Release hold, re-evaluate on next raw change
        }, cooldown)
      }
      return
    }

    // If we're still within the cooldown window, keep the held state
    if (heldState !== null) {
      const elapsed = Date.now() - heldSince
      const cooldown = COOLDOWN_MS[heldState]
      if (cooldown > 0 && elapsed < cooldown) {
        // Set timer for remaining cooldown
        const remaining = cooldown - elapsed
        cooldownTimerRef.current = window.setTimeout(() => {
          cooldownTimerRef.current = null
          setHeldState(null)
        }, remaining)
        return
      }
    }

    // Cooldown expired or no hold — use raw state
    setHeldState(null)
  }, [raw])

  // Determine effective state
  const effectiveState: RiskState = (() => {
    if (heldState !== null) {
      const elapsed = Date.now() - heldSince
      const cooldown = COOLDOWN_MS[heldState]
      if (cooldown > 0 && elapsed < cooldown && SEVERITY[heldState] > currentSeverity) {
        return heldState
      }
    }
    return currentRaw
  })()

  const isHeld = effectiveState !== currentRaw
  const holdRemainingMs = isHeld && heldState !== null
    ? Math.max(0, COOLDOWN_MS[heldState] - (Date.now() - heldSince))
    : 0

  // Build the adjusted assessment
  const assessment: FatigueAssessment | null = raw === null
    ? null
    : effectiveState === raw.state
      ? raw
      : { ...raw, state: effectiveState }

  return { assessment, isHeld, holdRemainingMs }
}
