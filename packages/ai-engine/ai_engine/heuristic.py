"""Rule-based heuristic fatigue classifier (ADITIVO, S1-17).

``HeuristicFatigueModel`` implements the ``FatigueModel`` port and replaces
``StubFatiguePredictor`` as the working predictor of the application. It is
the SEAM for the future trained model: a learned ``FatigueModel`` can be
substituted for this class through the same ``FatiguePipeline`` injection
without touching the pipeline, adapters or DI wiring.

Literature basis of the thresholds:

- Euro NCAP DSM 2023 (Driver State Monitoring test procedure): a
  microsleep episode is an eye closure < 3 s while ``sleep`` is an eye
  closure > 3 s. This module implements the escalation ladder up to
  ``CRITICAL_MICROSLEEP_MIN_S`` (2.0 s) and leaves the Euro NCAP
  ``SLEEP_MIN_S`` (3.0 s) escalon as documented future work (not implemented,
  no trained model yet — a 2.0 s closure is already a critical event for the
  dashboard).
- PERCLOS around 10% is a well-established driver-drowsiness corroborator;
  this module uses a conservative ``PERCLOS_CORROBORATION`` floor of 5% so a
  historical long closure does not need a full 10% to be corroborated.
- A long eye closure > 500 ms is a drowsiness marker (blink durations grow
  from the rested 100-200 ms to 300-500 ms under sleep deprivation); this is
  ``LONG_BLINK_MAX_S``.
- Blink rate below ``BLINK_RATE_LOW_CORROBORATION`` (8 blinks/min) is a
  corroborating fatigue signal on top of the closure evidence.

Design rule: the risk state must NOT depend only on closure duration. An
isolated long blink is a false-positive risk, so historical closures
(``longest_eye_closure_duration``) only escalate beyond a warning when the
closure is corroborated by PERCLOS, blink rate or yawn evidence. The ONLY
exception is an ACTIVE closure RIGHT NOW (``current_eye_closure_duration``),
which escalates immediately because the subject is currently not looking at
the road.
"""

from __future__ import annotations

from ai_engine.ports import FatigueModel
from ai_engine.types import (
    FatigueFeatures,
    FatigueLevel,
    FatigueResult,
    RiskState,
)

# -- validated threshold ladder (seconds) ------------------------------------

NORMAL_BLINK_MAX_S = 0.30
"""Upper bound of a normal blink (rested blinks last ~100-200 ms)."""

LONG_BLINK_MAX_S = 0.50
"""Upper bound of a long blink; > 500 ms is a drowsiness marker."""

FATIGUE_CLOSURE_MAX_S = 1.00
"""Closure duration that already warrants a fatigue warning."""

CRITICAL_MICROSLEEP_MIN_S = 2.00
"""Closure duration treated as an unambiguous critical microsleep.

Future Euro NCAP escalon: ``SLEEP_MIN_S = 3.00`` (eye closure > 3 s counts as
``sleep`` per Euro NCAP DSM). NOT implemented yet: without a trained model the
dashboard escalates to ``critical_microsleep`` at 2.0 s and keeps the Euro
NCAP sleep bucket as documented follow-up.
"""

# -- corroboration thresholds --------------------------------------------------

PERCLOS_CORROBORATION = 0.05
"""PERCLOS floor that corroborates a historical long closure."""

BLINK_RATE_LOW_CORROBORATION = 8.0
"""Blink rate at or below which a historical long closure is corroborated."""

# -- score calibration ---------------------------------------------------------

_BASE_SCORE: dict[RiskState, float] = {
    RiskState.normal: 12.0,
    RiskState.long_blink: 36.0,
    RiskState.fatigue_warning: 56.0,
    RiskState.microsleep_suspected: 76.0,
    RiskState.critical_microsleep: 93.0,
}

_SCORE_BANDS: dict[RiskState, tuple[float, float]] = {
    RiskState.normal: (0.0, 30.0),
    RiskState.long_blink: (30.0, 50.0),
    RiskState.fatigue_warning: (50.0, 70.0),
    RiskState.microsleep_suspected: (70.0, 88.0),
    RiskState.critical_microsleep: (88.0, 100.0),
}

_CORROBORATION_SCORE_NUDGE = 4.0
_YAWN_SCORE_NUDGE = 4.0

_LEVEL_BY_STATE: dict[RiskState, FatigueLevel] = {
    RiskState.normal: FatigueLevel.low,
    RiskState.long_blink: FatigueLevel.low,
    RiskState.fatigue_warning: FatigueLevel.medium,
    RiskState.microsleep_suspected: FatigueLevel.high,
    RiskState.critical_microsleep: FatigueLevel.high,
}

CONFIDENCE_BASELINE = 0.5
CONFIDENCE_CORROBORATION_BONUS = 0.2
CONFIDENCE_YAWN_BONUS = 0.15
CONFIDENCE_CAP = 0.95


def _score_for_state(
    state: RiskState, corroborated: bool, yawn_evidence: bool
) -> float:
    """Map a state to a 0-100 score, nudged up inside its band.

    Higher states sit in higher bands (``_SCORE_BANDS``); corroboration and
    yawn evidence nudge the score upward within the band WITHOUT crossing into
    the next state's band, so the score stays consistent with the state.
    """
    score = _BASE_SCORE[state]
    if corroborated:
        score += _CORROBORATION_SCORE_NUDGE
    if yawn_evidence:
        score += _YAWN_SCORE_NUDGE
    band_high = _SCORE_BANDS[state][1]
    return min(score, band_high - 1.0)


class HeuristicFatigueModel(FatigueModel):
    """Rule-based ``FatigueModel`` implementing the S1-17 risk-state ladder.

    Primary signal: eye closure duration. Corroborating signals: PERCLOS,
    blink rate and yawn evidence. A future trained model replaces this class
    via the same ``FatigueModel`` port without touching the pipeline or the
    DI wiring.
    """

    def predict(self, features: FatigueFeatures) -> FatigueResult:
        """Classify the features into a risk state and score them.

        The result is additive to the pre-S1-17 contract: ``timestamp`` is
        left None (the pipeline seals it), and the risk ``state`` always
        accompanies the level. Old feature dicts that lack the new additive
        fields classify as ``normal`` with a low score, keeping the model
        safe on degraded input.
        """
        corroborated = self._corroborated(features)
        yawn_evidence = self._has_yawn_evidence(features)
        state = self._classify(features, corroborated)
        return FatigueResult(
            score=_score_for_state(state, corroborated, yawn_evidence),
            level=_LEVEL_BY_STATE[state],
            confidence=_confidence_for(corroborated, yawn_evidence),
            timestamp=None,
            state=state,
        )

    @staticmethod
    def _has_yawn_evidence(features: FatigueFeatures) -> bool:
        """True when a yawn indicator is present (active or any completed)."""
        return features.current_yawn_duration > 0.0 or features.total_yawns > 0.0

    @classmethod
    def _corroborated(cls, features: FatigueFeatures) -> bool:
        """True when closure evidence is backed by a second signal.

        PERCLOS at or above the floor, a low blink rate, or any yawn evidence
        corroborates a HISTORICAL long closure so a single isolated long blink
        is not enough to escalate past the warning band.
        """
        return (
            features.perclos >= PERCLOS_CORROBORATION
            or features.blink_rate <= BLINK_RATE_LOW_CORROBORATION
            or cls._has_yawn_evidence(features)
        )

    @staticmethod
    def _classify(features: FatigueFeatures, corroborated: bool) -> RiskState:
        """Resolve the risk state from closure duration + corroboration.

        Active closure escalates immediately: the subject is closed RIGHT
        NOW. Historical closures require corroboration above the long-blink
        band to avoid a single long blink flagging a microsleep.
        """
        active = features.current_eye_closure_duration
        if active >= CRITICAL_MICROSLEEP_MIN_S:
            return RiskState.critical_microsleep
        if active >= FATIGUE_CLOSURE_MAX_S:
            return RiskState.microsleep_suspected

        longest = features.longest_eye_closure_duration
        if longest >= CRITICAL_MICROSLEEP_MIN_S:
            return RiskState.critical_microsleep
        if longest >= FATIGUE_CLOSURE_MAX_S:
            if corroborated:
                return RiskState.microsleep_suspected
            return RiskState.fatigue_warning
        if longest >= LONG_BLINK_MAX_S:
            if corroborated:
                return RiskState.fatigue_warning
            return RiskState.long_blink
        if longest >= NORMAL_BLINK_MAX_S:
            return RiskState.long_blink
        return RiskState.normal


def _confidence_for(corroborated: bool, yawn_evidence: bool) -> float:
    """Confidence in [0, 1]: 0.5 baseline, bumped by corroboration/yawns."""
    confidence = CONFIDENCE_BASELINE
    if corroborated:
        confidence += CONFIDENCE_CORROBORATION_BONUS
    if yawn_evidence:
        confidence += CONFIDENCE_YAWN_BONUS
    return min(confidence, CONFIDENCE_CAP)