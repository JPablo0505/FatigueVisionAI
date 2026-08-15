"""Unit tests for the sliding window and its FatigueFeatures aggregates."""

import pytest
from ai_engine import FatigueFeatures

from feature_extractor.window import (
    BiometricEvent,
    FeatureWindow,
    FrameMeasurements,
    WindowConfig,
)

FPS = 30.0


def _measurements(ear: float, mar: float) -> FrameMeasurements:
    """Open-eye measurements with neutral iris/pose, used across patterns."""
    return FrameMeasurements(
        ear=ear,
        mar=mar,
        iris_x=0.0,
        iris_y=0.0,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
    )


def _add_pattern(
    window: FeatureWindow, segments: list[tuple[float, float, int]]
) -> None:
    """Append (ear, mar, frame_count) segments at 30 fps starting at t=0."""
    _add_pattern_from(window, 0.0, segments)


def _add_pattern_from(
    window: FeatureWindow, start: float, segments: list[tuple[float, float, int]]
) -> None:
    """Append (ear, mar, frame_count) segments at 30 fps starting at ``start``."""
    timestamp = start
    for ear, mar, count in segments:
        for _ in range(count):
            window.add_frame(timestamp, _measurements(ear, mar))
            timestamp += 1.0 / FPS


def test_empty_window_aggregates_none() -> None:
    window = FeatureWindow()
    assert window.aggregate() is None


def test_single_frame_aggregates_valid_features() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    window.add_frame(0.0, _measurements(ear=0.4, mar=0.2))
    features = window.aggregate()
    assert isinstance(features, FatigueFeatures)
    assert features.ear == pytest.approx(0.4)
    assert features.mar == pytest.approx(0.2)
    assert features.perclos == pytest.approx(0.0)
    assert features.blink_rate == pytest.approx(0.0)
    assert features.window_seconds == 30.0


def test_perclos_half_closed_frames() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.5, 0.05, 30), (0.0, 0.05, 30)])
    features = window.aggregate()
    assert features.perclos == pytest.approx(0.5, abs=1e-12)
    assert features.ear == pytest.approx(0.25, abs=1e-12)


def test_blink_rate_counts_completed_closures_and_filters_glitches() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(
        window,
        [
            (0.5, 0.05, 15),
            (0.0, 0.05, 5),
            (0.0, 0.05, 1),
            (0.5, 0.05, 15),
            (0.0, 0.05, 5),
            (0.5, 0.05, 19),
        ],
    )
    features = window.aggregate()
    # Only the two 5-frame closures count as blinks; the 1-frame glitch is filtered.
    assert features.blink_rate == pytest.approx(4.0, abs=1e-9)


def test_eye_closure_duration_mean() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(
        window,
        [
            (0.5, 0.05, 10),
            (0.0, 0.05, 5),
            (0.5, 0.05, 10),
            (0.0, 0.05, 5),
            (0.5, 0.05, 10),
            (0.0, 0.05, 5),
            (0.5, 0.05, 15),
        ],
    )
    features = window.aggregate()
    assert features.eye_closure_duration == pytest.approx(1.0 / 6.0, abs=1e-9)
    assert features.blink_rate == pytest.approx(6.0, abs=1e-9)


def test_yawn_duration_mean() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(
        window,
        [
            (0.5, 0.05, 20),
            (0.5, 0.8, 10),
            (0.5, 0.05, 20),
            (0.5, 0.8, 10),
        ],
    )
    features = window.aggregate()
    assert features.yawn_duration == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_prune_drops_stale_frames() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    window.add_frame(0.0, _measurements(0.5, 0.05))
    window.add_frame(31.0, _measurements(0.0, 0.05))
    features = window.aggregate()
    # The t=0 frame is older than cutoff (31 - 30 = 1); only the closed frame survives.
    assert features.ear == pytest.approx(1e-4, abs=1e-12)
    assert features.perclos == pytest.approx(1.0)


def test_epsilon_floor_keeps_contract_valid() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    window.add_frame(0.0, _measurements(0.0, 0.05))
    window.add_frame(1.0 / FPS, _measurements(0.0, 0.05))
    features = window.aggregate()
    assert features.ear == pytest.approx(1e-4, abs=1e-12)
    assert features.perclos == pytest.approx(1.0)
    assert isinstance(features, FatigueFeatures)


def test_no_face_frames_do_not_pollute_metrics() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    window.add_frame(0.0, None)
    window.add_frame(1.0 / FPS, None)
    window.add_frame(2.0 / FPS, _measurements(0.5, 0.05))
    features = window.aggregate()
    assert features.ear == pytest.approx(0.5, abs=1e-12)
    assert features.perclos == pytest.approx(0.0)


def test_reset_clears_window() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.5, 0.05, 10)])
    assert window.aggregate() is not None
    window.reset()
    assert window.aggregate() is None


def test_current_yawn_duration_tracks_active_run_at_window_end() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.5, 0.05, 10), (0.5, 0.8, 15)])
    features = window.aggregate()
    # 15 trailing yawning frames at 30 fps => elapsed = (14 + 1) / 30 s.
    assert features.current_yawn_duration == pytest.approx(15.0 / FPS, abs=1e-9)
    # The active run is NOT completed, so the average stays over zero episodes.
    assert features.yawn_duration == pytest.approx(0.0)


def test_current_yawn_duration_zero_when_last_frame_not_yawning() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.5, 0.8, 10), (0.5, 0.05, 5)])
    features = window.aggregate()
    assert features.current_yawn_duration == pytest.approx(0.0)


def test_current_yawn_duration_zero_on_no_face_gap() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.5, 0.8, 10)])
    window.add_frame(10.0 / FPS, None)
    features = window.aggregate()
    # The last frame has no measurements: it breaks the run, so no active yawn.
    assert features.current_yawn_duration == pytest.approx(0.0)


def test_last_yawn_duration_is_last_completed_episode() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(
        window,
        [
            (0.5, 0.05, 10),
            (0.5, 0.8, 9),
            (0.5, 0.05, 10),
            (0.5, 0.8, 12),
            (0.5, 0.05, 5),
        ],
    )
    features = window.aggregate()
    # Two completed yawns; the last one spans 12 frames => 12 / 30 s.
    assert features.last_yawn_duration == pytest.approx(12.0 / FPS, abs=1e-9)
    assert features.yawn_duration == pytest.approx((9.0 / FPS + 12.0 / FPS) / 2.0, abs=1e-9)


def test_last_yawn_duration_zero_when_none_completed() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.5, 0.8, 12)])
    features = window.aggregate()
    assert features.last_yawn_duration == pytest.approx(0.0)


def test_mar_instant_is_last_metric_frame_mar() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.5, 0.2, 5), (0.5, 0.7, 5)])
    features = window.aggregate()
    assert features.mar_instant == pytest.approx(0.7, abs=1e-9)
    assert features.mar == pytest.approx(0.45, abs=1e-9)


def test_mar_instant_ignores_no_face_frames_after_it() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.5, 0.2, 5), (0.5, 0.8, 5)])
    window.add_frame(10.0 / FPS, None)
    window.add_frame(11.0 / FPS, None)
    features = window.aggregate()
    # mar_instant reflects the last frame WITH measurements, not the gap.
    assert features.mar_instant == pytest.approx(0.8, abs=1e-9)


def test_mar_instant_epsilon_floor_keeps_contract_valid() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.5, 0.0, 5)])
    features = window.aggregate()
    assert features.mar_instant == pytest.approx(1e-4, abs=1e-12)


def test_blink_event_queued_on_completion() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.0, 0.05, 5), (0.5, 0.05, 5)])
    events = window.drain_events()
    assert len(events) == 1
    assert events[0].kind == "blink"
    assert events[0].duration_seconds == pytest.approx(5.0 / FPS, abs=1e-9)
    assert events[0].timestamp_seconds == pytest.approx(4.0 / FPS, abs=1e-9)
    assert isinstance(events[0], BiometricEvent)
    assert window.drain_events() == []


def test_long_blink_event_on_borderline_closure() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    # 12 closed frames => 0.4 s, inside the 0.30-0.50 long-blink band.
    _add_pattern(window, [(0.0, 0.05, 12), (0.5, 0.05, 5)])
    events = window.drain_events()
    assert len(events) == 1
    assert events[0].kind == "long_blink"
    assert events[0].duration_seconds == pytest.approx(12.0 / FPS, abs=1e-9)


def test_eyes_closed_event_for_long_closure() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    # 30 closed frames => 1.0 s > 0.50 s: an eyes_closed event.
    _add_pattern(window, [(0.0, 0.05, 30), (0.5, 0.05, 5)])
    events = window.drain_events()
    assert len(events) == 1
    assert events[0].kind == "eyes_closed"
    assert events[0].duration_seconds == pytest.approx(1.0, abs=1e-9)


def test_yawn_event_queued_on_completion() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.5, 0.8, 10), (0.5, 0.05, 5)])
    events = window.drain_events()
    assert len(events) == 1
    assert events[0].kind == "yawn"
    assert events[0].duration_seconds == pytest.approx(10.0 / FPS, abs=1e-9)
    features = window.aggregate()
    assert features.total_yawns == 1
    assert features.longest_yawn_duration == pytest.approx(10.0 / FPS, abs=1e-9)


def test_single_frame_closure_emits_no_event() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.0, 0.05, 1), (0.5, 0.05, 5)])
    assert window.drain_events() == []


def test_active_run_emits_no_event_until_it_completes() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern_from(window, 0.0, [(0.0, 0.05, 5)])
    assert window.drain_events() == []
    # The first open frame completes the run, queuing the event now.
    _add_pattern_from(window, 5.0 / FPS, [(0.5, 0.05, 5)])
    events = window.drain_events()
    assert len(events) == 1
    assert events[0].kind == "blink"


def test_no_face_gap_completes_the_run_and_queues_its_event() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.0, 0.05, 10)])
    window.add_frame(10.0 / FPS, None)
    features = window.aggregate()
    # The gap broke the run: no active closure, and a long_blink event queued.
    assert features.current_eye_closure_duration == pytest.approx(0.0)
    events = window.drain_events()
    assert len(events) == 1
    assert events[0].kind == "long_blink"


def test_events_keep_chronological_order() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(
        window,
        [
            (0.0, 0.05, 5),
            (0.5, 0.05, 5),
            (0.0, 0.05, 5),
            (0.5, 0.05, 5),
        ],
    )
    events = window.drain_events()
    assert [e.kind for e in events] == ["blink", "blink"]
    assert events[0].timestamp_seconds < events[1].timestamp_seconds
    features = window.aggregate()
    assert features.total_blinks == 2


def test_event_buffer_caps_at_max_keeping_newest() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    timestamp = 0.0
    for _ in range(300):
        for _ in range(5):
            window.add_frame(timestamp, _measurements(0.0, 0.05))
            timestamp += 1.0 / FPS
        for _ in range(5):
            window.add_frame(timestamp, _measurements(0.5, 0.05))
            timestamp += 1.0 / FPS
    events = window.drain_events()
    assert len(events) == 256
    assert all(event.kind == "blink" for event in events)
    assert all(event.duration_seconds == pytest.approx(5.0 / FPS) for event in events)


def test_session_totals_survive_window_pruning() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern_from(window, 0.0, [(0.0, 0.05, 5), (0.5, 0.05, 5)])
    # Jump far past the window: the first blink's frames get pruned.
    window.add_frame(40.0, _measurements(0.5, 0.05))
    _add_pattern_from(window, 40.0, [(0.0, 0.05, 5), (0.5, 0.05, 5)])
    features = window.aggregate()
    # Session counters keep BOTH blinks; the window-scoped rate sees only one.
    assert features.total_blinks == 2
    assert features.longest_eye_closure_duration == pytest.approx(5.0 / FPS, abs=1e-9)
    assert features.blink_rate == pytest.approx(2.0, abs=1e-9)


def test_reset_clears_events_and_session_counters() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.0, 0.05, 5), (0.5, 0.05, 5)])
    window.reset()
    assert window.aggregate() is None
    assert window.drain_events() == []
    _add_pattern(window, [(0.0, 0.05, 5), (0.5, 0.05, 5)])
    features = window.aggregate()
    assert features.total_blinks == 1
    assert len(window.drain_events()) == 1


def test_current_eye_closure_tracks_active_run_at_window_end() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.5, 0.05, 10), (0.0, 0.05, 15)])
    features = window.aggregate()
    # 15 trailing closed frames at 30 fps => elapsed = (14 + 1) / 30 s.
    assert features.current_eye_closure_duration == pytest.approx(15.0 / FPS, abs=1e-9)
    # The active run is not completed: no event, no session longest yet.
    assert window.drain_events() == []
    assert features.longest_eye_closure_duration == pytest.approx(0.0)


def test_current_eye_closure_zero_when_last_frame_open() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.0, 0.05, 10), (0.5, 0.05, 5)])
    features = window.aggregate()
    assert features.current_eye_closure_duration == pytest.approx(0.0)
    # The completed 10-frame run feeds the session longest.
    assert features.longest_eye_closure_duration == pytest.approx(10.0 / FPS, abs=1e-9)


def test_ear_instant_is_last_metric_frame_ear() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.5, 0.05, 5), (0.3, 0.05, 5)])
    features = window.aggregate()
    assert features.ear_instant == pytest.approx(0.3, abs=1e-9)
    assert features.ear == pytest.approx(0.4, abs=1e-9)


def test_ear_instant_epsilon_floor_keeps_contract_valid() -> None:
    window = FeatureWindow(WindowConfig(window_seconds=30.0))
    _add_pattern(window, [(0.0, 0.05, 5)])
    features = window.aggregate()
    assert features.ear_instant == pytest.approx(1e-4, abs=1e-12)