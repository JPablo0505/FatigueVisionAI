"""Sliding time window that aggregates per-frame measurements.

The window is the only stateful piece of the extractor: biometric
measurements (PERCLOS, blink count, episode durations) need a temporal
context that a single frame cannot provide. Frames older than the
configured window are pruned on insertion; ``aggregate`` reduces the
in-window frames into the exact ``FatigueFeatures`` contract validated by
the AI Engine without mutating the window.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite

from ai_engine.types import FatigueFeatures, HeadPose, IrisDirection

_METRICS_EPSILON = 1e-4
"""Floor applied to strictly-positive aggregated measurements (ear, mar).

The AI Engine contract is strict on ear and mar being positive. When a
window's raw mean is 0.0 (e.g. every frame closed or degenerate) this
documented epsilon is emitted instead of an invalid feature set.
"""

_MIN_BLINK_FRAMES = 2
"""Minimum consecutive closed frames for a closure to count as a blink.

At typical sample rates a single-frame "closure" is detection noise rather
than a real blink; requiring a minimum length filters those glitches out.
"""

_NORMAL_BLINK_MAX_S = 0.30
"""Upper bound of a normal blink for biometric-event classification.

Mirror of ``ai_engine.heuristic.NORMAL_BLINK_MAX_S``: the extractor's event
classification must stay 1:1 with the heuristic's risk-state ladder.
"""

_LONG_BLINK_MAX_S = 0.50
"""Upper bound of a long blink for biometric-event classification.

Mirror of ``ai_engine.heuristic.LONG_BLINK_MAX_S`` (the >500 ms drowsiness
marker). Exactly 0.50 s still classifies as ``long_blink`` (consistent with
the heuristic band); anything longer is an ``eyes_closed`` event.
"""

_MAX_BIOMETRIC_EVENTS = 256
"""Cap of the append-only biometric event buffer (oldest dropped first)."""


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive ``[low, high]`` interval."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def _finite(value: float, default: float) -> float:
    """Return ``value`` when finite, else ``default`` (input sanitation)."""
    return value if isfinite(value) else default


def _clamp_duration(value: float) -> float:
    """Clamp a duration into the contract's ``[0, 20]`` seconds range.

    Non-finite values collapse to 0.0. Real capture data never approaches
    the cap; the clamp exists defensively for synthetic/extreme inputs.
    """
    if not isfinite(value):
        return 0.0
    return min(20.0, max(0.0, value))


def _mean(values: list[float]) -> float:
    """Arithmetic mean of ``values``; empty input yields 0.0."""
    if not values:
        return 0.0
    return sum(values) / len(values)


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Configuration of a sliding time window."""

    window_seconds: float = 5.0
    ear_closed_threshold: float = 0.25
    mar_yawn_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.window_seconds <= 86_400.0:
            raise ValueError(
                f"window_seconds must be in (0, 86400], got {self.window_seconds}"
            )


@dataclass(frozen=True, slots=True)
class FrameMeasurements:
    """Raw per-frame measurements fed to the window by the extractor.

    The binarized states (eyes closed, yawning) are derived internally from
    the configured thresholds, so the window owns the threshold semantics.
    """

    ear: float
    mar: float
    iris_x: float
    iris_y: float
    yaw: float
    pitch: float
    roll: float


@dataclass(frozen=True, slots=True)
class _WindowFrame:
    """A recorded frame; ``None`` measurements mean no face was available."""

    timestamp: float
    measurements: FrameMeasurements | None
    eyes_closed: bool
    yawning: bool


@dataclass(frozen=True, slots=True)
class _Episode:
    """A completed run (eye closure or mouth-opening) of consecutive frames."""

    start: _WindowFrame
    end: _WindowFrame
    frame_count: int

    def duration(self, frame_period: float) -> float:
        """Duration in seconds; mid-run frame times are spaced by ``frame_period``."""
        return (self.end.timestamp - self.start.timestamp) + frame_period


def _condition_closed(frame: _WindowFrame) -> bool:
    """True for metric frames whose eyes are closed."""
    return frame.measurements is not None and frame.eyes_closed


def _condition_yawning(frame: _WindowFrame) -> bool:
    """True for metric frames whose mouth is open (yawning)."""
    return frame.measurements is not None and frame.yawning


def _closure_kind(duration: float) -> str:
    """Classify a completed closure run by its duration.

    ``< 0.30`` s is a normal blink, ``0.30-0.50`` s a long blink and anything
    longer an ``eyes_closed`` event (mirrors the heuristic thresholds).
    """
    if duration < _NORMAL_BLINK_MAX_S:
        return "blink"
    if duration <= _LONG_BLINK_MAX_S:
        return "long_blink"
    return "eyes_closed"


@dataclass(frozen=True, slots=True)
class BiometricEvent:
    """A completed biometric episode recorded by the window (ADITIVO, S1-17).

    ``kind`` is one of ``"blink"`` | ``"long_blink"`` | ``"eyes_closed"`` |
    ``"yawn"``. ``timestamp_seconds`` is the epoch (seconds) of the LAST frame
    of the completed run; ``duration_seconds`` its total duration. Events are
    queued when a run completes and drained via ``drain_events`` so the
    backend can broadcast them as ``biometric.event`` envelopes.
    """

    kind: str
    timestamp_seconds: float
    duration_seconds: float


class FeatureWindow:
    """Sliding window of per-frame measurements and their aggregates."""

    def __init__(self, config: WindowConfig | None = None) -> None:
        """Build an empty window; default config spans 30 seconds."""
        self._config = config if config is not None else WindowConfig()
        self._frames: list[_WindowFrame] = []
        # SESSION state (ADITIVO, S1-17): cleared by ``reset``, NOT pruned
        # like the sliding window. Events and counters accumulate for the
        # whole session so the realtime risk dashboard sees recent events and
        # session totals even after their frames drifted out of the window.
        self._events: list[BiometricEvent] = []
        self._total_blinks = 0
        self._total_yawns = 0
        self._longest_eye_closure_duration = 0.0
        self._longest_yawn_duration = 0.0
        self._closure_start: _WindowFrame | None = None
        self._closure_count = 0
        self._yawn_start: _WindowFrame | None = None
        self._yawn_count = 0

    @property
    def config(self) -> WindowConfig:
        """The window configuration (read-only)."""
        return self._config

    def add_frame(
        self, timestamp_seconds: float, measurements: FrameMeasurements | None
    ) -> None:
        """Record a frame; ``None`` measurements mean no face was available.

        Frames without measurements keep the window advancing (so pruning
        stays continuous) but contribute nothing to the aggregates and break
        episode runs, mirroring a real tracking gap.

        Session bookkeeping (ADITIVO, S1-17) runs here: when a closure or yawn
        run completes its ``BiometricEvent`` is queued and the session counters
        update, independent of the sliding-window pruning.
        """
        if measurements is None:
            frame = _WindowFrame(timestamp_seconds, None, False, False)
        else:
            eyes_closed = measurements.ear < self._config.ear_closed_threshold
            yawning = measurements.mar > self._config.mar_yawn_threshold
            frame = _WindowFrame(timestamp_seconds, measurements, eyes_closed, yawning)

        previous = self._frames[-1] if self._frames else None
        self._frames.append(frame)
        self._prune(self._config.window_seconds)
        frame_period = self._frame_period()

        # Closure run: completes when a closed frame is followed by a
        # non-closed one (a ``None``-measurement frame breaks runs exactly
        # like ``_episodes`` does). ``previous`` is guaranteed present while a
        # run is active (runs only start once frames exist); the None check is
        # purely defensive for the type checker.
        if (
            self._closure_start is not None
            and previous is not None
            and not _condition_closed(frame)
        ):
            self._complete_closure(previous, frame_period)
        if _condition_closed(frame):
            if self._closure_start is None:
                self._closure_start = frame
                self._closure_count = 1
            else:
                self._closure_count += 1
        else:
            self._closure_start = None
            self._closure_count = 0

        # Yawn run: same transition logic, independent of the closure run (a
        # frame can be both closed and yawning at once).
        if (
            self._yawn_start is not None
            and previous is not None
            and not _condition_yawning(frame)
        ):
            self._complete_yawn(previous, frame_period)
        if _condition_yawning(frame):
            if self._yawn_start is None:
                self._yawn_start = frame
                self._yawn_count = 1
            else:
                self._yawn_count += 1
        else:
            self._yawn_start = None
            self._yawn_count = 0

    def _complete_closure(self, end_frame: _WindowFrame, frame_period: float) -> None:
        """Close the active closure run; queue its event and update session stats.

        The duration uses the same math as ``_Episode.duration``: the last
        closed frame's timestamp minus the run start, plus the median frame
        period. Runs below ``_MIN_BLINK_FRAMES`` are detection noise and
        produce no event.
        """
        start = self._closure_start
        count = self._closure_count
        self._closure_start = None
        self._closure_count = 0
        if start is None or count < _MIN_BLINK_FRAMES:
            return
        duration = _clamp_duration(
            (end_frame.timestamp - start.timestamp) + frame_period
        )
        self._append_event(
            BiometricEvent(
                kind=_closure_kind(duration),
                timestamp_seconds=end_frame.timestamp,
                duration_seconds=duration,
            )
        )
        self._total_blinks += 1
        self._longest_eye_closure_duration = max(self._longest_eye_closure_duration, duration)

    def _complete_yawn(self, end_frame: _WindowFrame, frame_period: float) -> None:
        """Close the active yawn run; queue its event and update session stats.

        The same minimum-frame filter as closures avoids single-frame yawn
        noise in the event stream.
        """
        start = self._yawn_start
        count = self._yawn_count
        self._yawn_start = None
        self._yawn_count = 0
        if start is None or count < _MIN_BLINK_FRAMES:
            return
        duration = _clamp_duration(
            (end_frame.timestamp - start.timestamp) + frame_period
        )
        self._append_event(
            BiometricEvent(
                kind="yawn",
                timestamp_seconds=end_frame.timestamp,
                duration_seconds=duration,
            )
        )
        self._total_yawns += 1
        self._longest_yawn_duration = max(self._longest_yawn_duration, duration)

    def _append_event(self, event: BiometricEvent) -> None:
        """Append one event, dropping the oldest when the buffer is full."""
        self._events.append(event)
        overflow = len(self._events) - _MAX_BIOMETRIC_EVENTS
        if overflow > 0:
            del self._events[:overflow]

    def drain_events(self) -> list[BiometricEvent]:
        """Return and clear the queued biometric events, preserving order.

        Each event is returned exactly once; a second call yields an empty
        list until new runs complete.
        """
        events = list(self._events)
        self._events = []
        return events

    def _prune(self, window_seconds: float) -> None:
        """Drop frames older than ``window_seconds`` relative to the newest."""
        if not self._frames:
            return
        cutoff = self._frames[-1].timestamp - window_seconds
        self._frames = [frame for frame in self._frames if frame.timestamp >= cutoff]

    def _frame_period(self) -> float:
        """Median consecutive timestamp delta (median-robust sample period)."""
        times = [frame.timestamp for frame in self._frames]
        if len(times) < 2:
            return 0.0
        diffs = sorted(b - a for a, b in pairwise(times) if b > a)
        if not diffs:
            return 0.0
        return diffs[len(diffs) // 2]

    def _episodes(self, condition: Callable[[_WindowFrame], bool]) -> list[_Episode]:
        """Completed runs of consecutive frames matching ``condition``.

        A run is completed when a matching frame is followed by a
        non-matching one; a run still active at the last frame is not
        counted (its true duration is unknown). ``None``-measurement frames
        break runs exactly like a tracking gap.
        """
        episodes: list[_Episode] = []
        start: _WindowFrame | None = None
        last: _WindowFrame | None = None
        count = 0
        for frame in self._frames:
            if condition(frame):
                if start is None:
                    start = frame
                last = frame
                count += 1
            elif start is not None:
                episodes.append(
                    _Episode(start=start, end=last or start, frame_count=count)
                )
                start = None
                last = None
                count = 0
        return episodes

    def aggregate(self) -> FatigueFeatures | None:
        """Reduce in-window frames to ``FatigueFeatures``, or None when empty.

        Only frames that carried measurements contribute; ``None`` frames
        only advance the window. Blink counts and episode durations are
        recomputed from the frames currently in the window, so pruning is
        always reflected (a blink that drifted out of the window no longer
        counts). The SESSION fields (``total_blinks``, ``total_yawns`` and
        the longest durations) are NOT window-scoped: they accumulate for the
        whole session and are cleared only by ``reset``. The window state is
        left untouched.
        """
        metric_frames = [f for f in self._frames if f.measurements is not None]
        if not metric_frames:
            return None
        metric = [f.measurements for f in metric_frames if f.measurements is not None]
        count = len(metric_frames)

        closed_count = sum(1 for frame in metric_frames if frame.eyes_closed)
        perclos = closed_count / count

        ear_mean = sum(m.ear for m in metric) / count
        mar_mean = sum(m.mar for m in metric) / count
        iris_x_mean = sum(m.iris_x for m in metric) / count
        iris_y_mean = sum(m.iris_y for m in metric) / count
        yaw_mean = sum(m.yaw for m in metric) / count
        pitch_mean = sum(m.pitch for m in metric) / count
        roll_mean = sum(m.roll for m in metric) / count

        frame_period = self._frame_period()
        closure_episodes = self._episodes(_condition_closed)
        yawn_episodes = self._episodes(_condition_yawning)

        blinks = [e for e in closure_episodes if e.frame_count >= _MIN_BLINK_FRAMES]
        blink_rate = len(blinks) * 60.0 / self._config.window_seconds
        eye_closure_duration = _clamp_duration(
            _mean([episode.duration(frame_period) for episode in closure_episodes])
        )
        yawn_duration = _clamp_duration(
            _mean([episode.duration(frame_period) for episode in yawn_episodes])
        )

        ear = _finite(ear_mean, _METRICS_EPSILON)
        mar = _finite(mar_mean, _METRICS_EPSILON)
        if ear <= 0.0:
            ear = _METRICS_EPSILON
        if mar <= 0.0:
            mar = _METRICS_EPSILON

        iris_x = _clamp(_finite(iris_x_mean, 0.0), -1.0, 1.0)
        iris_y = _clamp(_finite(iris_y_mean, 0.0), -1.0, 1.0)

        # Live indicators (ADITIVO): realtime dashboard values that DO NOT
        # change ``yawn_duration`` (the average of COMPLETED yawns). The
        # ``mar`` above is the per-window mean; ``mar_instant`` is the raw mar
        # of the LAST metric frame, floored to the same epsilon so the
        # contract stays strictly positive.
        mar_instant = metric[-1].mar
        if mar_instant <= 0.0:
            mar_instant = _METRICS_EPSILON

        # Realtime EAR: raw ear of the LAST metric frame, floored to the same
        # epsilon so the strictly-positive contract holds.
        ear_instant = metric[-1].ear
        if ear_instant <= 0.0:
            ear_instant = _METRICS_EPSILON

        # Current yawn: duration of the ACTIVE run when the window's last
        # frame is still yawning. ``_condition_yawning`` treats frames without
        # measurements as non-yawning, so a tracking gap breaks the run.
        current_yawn_duration = 0.0
        if self._frames and _condition_yawning(self._frames[-1]):
            active_start = self._frames[-1]
            for frame in reversed(self._frames):
                if not _condition_yawning(frame):
                    break
                active_start = frame
            current_yawn_duration = _clamp_duration(
                (self._frames[-1].timestamp - active_start.timestamp) + frame_period
            )

        # Current eye closure: duration of the ACTIVE closure run when the
        # window's last frame still has the eyes closed; mirrors the active
        # yawn computation above.
        current_eye_closure_duration = 0.0
        if self._frames and _condition_closed(self._frames[-1]):
            active_start = self._frames[-1]
            for frame in reversed(self._frames):
                if not _condition_closed(frame):
                    break
                active_start = frame
            current_eye_closure_duration = _clamp_duration(
                (self._frames[-1].timestamp - active_start.timestamp) + frame_period
            )

        # Last yawn: duration of the last COMPLETED episode (in window order).
        last_yawn_duration = 0.0
        if yawn_episodes:
            last_yawn_duration = _clamp_duration(
                yawn_episodes[-1].duration(frame_period)
            )

        return FatigueFeatures(
            ear=ear,
            mar=mar,
            perclos=_finite(perclos, 0.0),
            blink_rate=_finite(blink_rate, 0.0),
            head_pose=HeadPose(
                yaw=_finite(yaw_mean, 0.0),
                pitch=_finite(pitch_mean, 0.0),
                roll=_finite(roll_mean, 0.0),
            ),
            eye_closure_duration=eye_closure_duration,
            yawn_duration=yawn_duration,
            current_yawn_duration=current_yawn_duration,
            last_yawn_duration=last_yawn_duration,
            mar_instant=mar_instant,
            ear_instant=ear_instant,
            current_eye_closure_duration=current_eye_closure_duration,
            longest_eye_closure_duration=self._longest_eye_closure_duration,
            longest_yawn_duration=self._longest_yawn_duration,
            total_blinks=float(self._total_blinks),
            total_yawns=float(self._total_yawns),
            iris_direction=IrisDirection(x=iris_x, y=iris_y),
            window_seconds=self._config.window_seconds,
        )

    def reset(self) -> None:
        """Clear all window state, including events and session counters."""
        self._frames = []
        self._events = []
        self._total_blinks = 0
        self._total_yawns = 0
        self._longest_eye_closure_duration = 0.0
        self._longest_yawn_duration = 0.0
        self._closure_start = None
        self._closure_count = 0
        self._yawn_start = None
        self._yawn_count = 0


__all__ = ["BiometricEvent", "FeatureWindow", "FrameMeasurements", "WindowConfig"]