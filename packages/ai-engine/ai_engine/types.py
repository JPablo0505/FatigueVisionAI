"""Pure data contracts exchanged by the AI Engine.

These dataclasses are the high-level feature contract of the engine: they
carry no pixels, no landmarks and no media dependency (stdlib only). All
values are validated on construction so a ``FatigueFeatures`` instance is
always safe to pass to any ``FatigueModel`` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from ai_engine.errors import InvalidFeaturesError


class FatigueLevel(Enum):
    """Discrete fatigue level produced by the engine."""

    low = "low"
    medium = "medium"
    high = "high"


class RiskState(Enum):
    """Realtime risk state of the subject (ADITIVO, S1-17).

    Vocabulary coupled 1:1 with the ``RiskState`` type of
    ``@fatiguevision/contracts``: a change here must be reflected in the
    TypeScript type and vice versa. The state is produced by the heuristic
    classifier and rides in ``FatigueResult.state``; the discrete levels
    (``FatigueLevel``) are derived from this state.
    """

    normal = "normal"
    long_blink = "long_blink"
    fatigue_warning = "fatigue_warning"
    microsleep_suspected = "microsleep_suspected"
    critical_microsleep = "critical_microsleep"


@dataclass(frozen=True, slots=True)
class HeadPose:
    """Head orientation angles in radians.

    Angles follow the common Euler convention with yaw around the vertical
    axis, pitch around the horizontal axis and roll around the viewing axis,
    all expressed in radians.
    """

    yaw: float
    pitch: float
    roll: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-clean representation of this pose."""
        return {"yaw": self.yaw, "pitch": self.pitch, "roll": self.roll}


@dataclass(frozen=True, slots=True)
class IrisDirection:
    """Normalized gaze direction of the iris.

    Each component is expected to be normalized to the -1..1 range.
    """

    x: float
    y: float

    def __post_init__(self) -> None:
        for name, value in (("x", self.x), ("y", self.y)):
            if not isfinite(value) or not -1.0 <= value <= 1.0:
                raise InvalidFeaturesError(
                    f"IrisDirection.{name} must be normalized to [-1, 1], got {value}"
                )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-clean representation of this direction."""
        return {"x": self.x, "y": self.y}


def _require_finite_range(
    value: float,
    name: str,
    low: float,
    high: float,
    *,
    strict_low: bool = False,
    strict_high: bool = False,
) -> None:
    """Raise InvalidFeaturesError unless ``value`` falls within the range.

    ``strict_*`` flags make the corresponding bound exclusive. ``float("nan")``
    and infinities are always rejected.
    """
    if not isfinite(value):
        raise InvalidFeaturesError(f"{name} must be a finite number, got {value}")
    too_low = value <= low if strict_low else value < low
    too_high = value >= high if strict_high else value > high
    if too_low or too_high:
        lower = "(" if strict_low else "["
        upper = ")" if strict_high else "]"
        raise InvalidFeaturesError(
            f"{name} must be in {lower}{low}, {high}{upper}, got {value}"
        )


_MAR_MEASUREMENT_EPSILON = 1e-4
"""Floor for the strictly-positive MAR indicators (``mar``, ``mar_instant``).

The Feature Extractor emits a documented epsilon instead of 0.0 for degenerate
windows, so a strictly-positive ``mar_instant`` is the contract's expectation.
The default mirrors that floor: a plain ``FatigueFeatures(...)`` without
``mar_instant`` stays valid, while an explicit 0.0 (or negative) is rejected
exactly like ``mar``.
"""

_EAR_MEASUREMENT_EPSILON = 1e-4
"""Floor for the strictly-positive EAR indicator (``ear_instant``).

``ear`` is validated strictly positive, so the realtime ``ear_instant`` follows
the same rule: the Feature Extractor floors a degenerate 0.0 to this epsilon,
and an explicit 0.0 (or negative) here is rejected exactly like ``ear``.
"""


@dataclass(frozen=True, slots=True)
class FatigueFeatures:
    """High-level fatigue features observed over a sliding time window.

    The engine receives this contract only; it knows nothing about cameras,
    frames or landmark extraction. Values are the per-window aggregate
    features produced upstream by the Vision Engine. The live fields
    (``current_yawn_duration``, ``last_yawn_duration``, ``mar_instant``) are
    ADDITIVE indicators for the realtime dashboard; they never alter the
    ``yawn_duration`` average semantics.

    The S1-17 realtime fields (``ear_instant``,
    ``current_eye_closure_duration``, ``longest_*``, ``total_*``) are ALSO
    additive. ``current_eye_closure_duration`` is the ACTIVE run at the last
    frame (mirror of ``current_yawn_duration``); ``longest_*`` and ``total_*``
    are SESSION counters since the last extractor reset, NOT window-scoped —
    they accumulate across the whole streaming session while the per-window
    aggregates (``eye_closure_duration``, ``yawn_duration``, ``blink_rate``)
    describe only the sliding window.
    """

    ear: float
    mar: float
    perclos: float
    blink_rate: float
    head_pose: HeadPose
    eye_closure_duration: float
    yawn_duration: float
    iris_direction: IrisDirection
    window_seconds: float
    # Live indicators (ADITIVO): they complement the per-window aggregates for
    # the realtime dashboard WITHOUT changing ``yawn_duration`` semantics (the
    # average of COMPLETED yawns). Defaults keep existing constructors valid.
    current_yawn_duration: float = 0.0
    last_yawn_duration: float = 0.0
    mar_instant: float = _MAR_MEASUREMENT_EPSILON
    # Realtime risk + session indicators (ADITIVO, S1-17): complement the
    # per-window aggregates for the risk-state classifier and the realtime
    # dashboard. Defaults keep existing constructors and old feature dicts
    # valid; old servers/clients that omit them keep working unchanged.
    ear_instant: float = _EAR_MEASUREMENT_EPSILON
    current_eye_closure_duration: float = 0.0
    longest_eye_closure_duration: float = 0.0
    longest_yawn_duration: float = 0.0
    total_blinks: float = 0.0
    total_yawns: float = 0.0

    def __post_init__(self) -> None:
        _require_finite_range(self.ear, "ear", 0.0, 1.0, strict_low=True)
        _require_finite_range(self.mar, "mar", 0.0, 1.5, strict_low=True)
        _require_finite_range(self.perclos, "perclos", 0.0, 1.0)
        _require_finite_range(self.blink_rate, "blink_rate", 0.0, 300.0)
        _require_finite_range(
            self.eye_closure_duration, "eye_closure_duration", 0.0, 20.0
        )
        _require_finite_range(self.yawn_duration, "yawn_duration", 0.0, 20.0)
        _require_finite_range(
            self.window_seconds, "window_seconds", 0.0, 86_400.0, strict_low=True
        )
        _require_finite_range(
            self.current_yawn_duration, "current_yawn_duration", 0.0, 20.0
        )
        _require_finite_range(
            self.last_yawn_duration, "last_yawn_duration", 0.0, 20.0
        )
        _require_finite_range(
            self.mar_instant, "mar_instant", 0.0, 1.5, strict_low=True
        )
        _require_finite_range(self.ear_instant, "ear_instant", 0.0, 1.0, strict_low=True)
        _require_finite_range(
            self.current_eye_closure_duration,
            "current_eye_closure_duration",
            0.0,
            20.0,
        )
        _require_finite_range(
            self.longest_eye_closure_duration,
            "longest_eye_closure_duration",
            0.0,
            20.0,
        )
        _require_finite_range(
            self.longest_yawn_duration, "longest_yawn_duration", 0.0, 20.0
        )
        _require_finite_range(self.total_blinks, "total_blinks", 0.0, 1_000_000.0)
        _require_finite_range(self.total_yawns, "total_yawns", 0.0, 1_000_000.0)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-clean representation of these features."""
        return {
            "ear": self.ear,
            "mar": self.mar,
            "mar_instant": self.mar_instant,
            "ear_instant": self.ear_instant,
            "perclos": self.perclos,
            "blink_rate": self.blink_rate,
            "head_pose": self.head_pose.to_dict(),
            "eye_closure_duration": self.eye_closure_duration,
            "yawn_duration": self.yawn_duration,
            "current_yawn_duration": self.current_yawn_duration,
            "last_yawn_duration": self.last_yawn_duration,
            "current_eye_closure_duration": self.current_eye_closure_duration,
            "longest_eye_closure_duration": self.longest_eye_closure_duration,
            "longest_yawn_duration": self.longest_yawn_duration,
            "total_blinks": self.total_blinks,
            "total_yawns": self.total_yawns,
            "iris_direction": self.iris_direction.to_dict(),
            "window_seconds": self.window_seconds,
        }


def default_fatigue_level(score: float) -> FatigueLevel:
    """Map a fatigue score to a level using the default business rule.

    Default policy: ``score < 33`` -> low, ``score < 66`` -> medium, else
    high. This is a calibratable default; real model implementations may
    override the level with their own mapping.
    """
    if score < 33.0:
        return FatigueLevel.low
    if score < 66.0:
        return FatigueLevel.medium
    return FatigueLevel.high


@dataclass(frozen=True, slots=True)
class FatigueResult:
    """Output contract of a fatigue assessment."""

    score: float
    level: FatigueLevel
    confidence: float
    timestamp: float | None = None
    # Realtime risk state (ADITIVO, S1-17): added AFTER ``timestamp`` so
    # existing constructors and older consumers keep working. ``normal`` is
    # the additive default for results produced by predictors that predate it.
    state: RiskState = RiskState.normal

    def __post_init__(self) -> None:
        _require_finite_range(self.score, "score", 0.0, 100.0)
        _require_finite_range(self.confidence, "confidence", 0.0, 1.0)
        if self.timestamp is not None:
            _require_finite_range(self.timestamp, "timestamp", 0.0, 9_223_372_036_854_775.0)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-clean representation of this result.

        When ``timestamp`` is None it is serialized as JSON null so the
        pipeline can seal it later.
        """
        return {
            "score": self.score,
            "level": self.level.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "state": self.state.value,
        }