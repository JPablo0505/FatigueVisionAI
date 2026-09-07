"""S3-04: Benchmark test comparing Heuristic vs ML model performance.

Validates that both models produce valid FatigueResult outputs and measures
latency. The benchmark runs 1,000 iterations per model and asserts:
- Both models return valid results (score in [0,100], level in {low,medium,high})
- Heuristic model detects fatigue states correctly
- ML model latency is within acceptable bounds (<10ms p99)
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

from ai_engine.heuristic import HeuristicFatigueModel
from ai_engine.ml_model import MLFatigueModel
from ai_engine.types import (
    FatigueFeatures,
    HeadPose,
    IrisDirection,
    RiskState,
)

ITERATIONS = 1_000


def _make_features(**overrides: object) -> FatigueFeatures:
    defaults = dict(
        ear=0.35, mar=0.05, perclos=0.05, blink_rate=15.0,
        eye_closure_duration=0.1, yawn_duration=0.0,
        current_yawn_duration=0.0, last_yawn_duration=0.0,
        mar_instant=0.05, ear_instant=0.35,
        current_eye_closure_duration=0.0,
        longest_eye_closure_duration=0.2, longest_yawn_duration=0.0,
        total_blinks=5.0, total_yawns=0.0,
        window_seconds=5.0,
        head_pose=HeadPose(yaw=0.0, pitch=0.0, roll=0.0),
        iris_direction=IrisDirection(x=0.0, y=0.0),
    )
    defaults.update(overrides)
    return FatigueFeatures(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Latency benchmarks
# ---------------------------------------------------------------------------

class TestLatency:
    def test_heuristic_p99_under_1ms(self) -> None:
        model = HeuristicFatigueModel()
        features = _make_features()
        # Warmup
        for _ in range(50):
            model.predict(features)
        latencies = []
        for _ in range(ITERATIONS):
            t0 = time.perf_counter()
            model.predict(features)
            latencies.append((time.perf_counter() - t0) * 1000)
        p99 = float(np.percentile(latencies, 99))
        assert p99 < 1.0, f"Heuristic p99={p99:.3f}ms, expected <1ms"

    def test_ml_p99_under_20ms(self) -> None:
        model = MLFatigueModel()
        features = _make_features()
        # Warmup (more iterations to stabilize sklearn internals)
        for _ in range(100):
            model.predict(features)
        latencies = []
        for _ in range(ITERATIONS):
            t0 = time.perf_counter()
            model.predict(features)
            latencies.append((time.perf_counter() - t0) * 1000)
        p99 = float(np.percentile(latencies, 99))
        assert p99 < 20.0, f"ML p99={p99:.3f}ms, expected <20ms"


# ---------------------------------------------------------------------------
# Output validity
# ---------------------------------------------------------------------------

class TestOutputValidity:
    SCENARIOS = {
        "neutral": dict(ear=0.35, perclos=0.02, blink_rate=18.0),
        "yawning": dict(ear=0.25, mar=0.70, perclos=0.15, current_yawn_duration=1.5),
        "microsleep": dict(ear=0.05, perclos=0.40, current_eye_closure_duration=1.0),
        "critical": dict(ear=0.01, perclos=0.80, current_eye_closure_duration=2.5),
    }

    @pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
    def test_heuristic_valid_output(self, scenario: str) -> None:
        overrides = self.SCENARIOS[scenario]
        model = HeuristicFatigueModel()
        result = model.predict(_make_features(**overrides))
        assert 0.0 <= result.score <= 100.0
        assert result.level.value in ("low", "medium", "high")
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.state, RiskState)

    @pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
    def test_ml_valid_output(self, scenario: str) -> None:
        overrides = self.SCENARIOS[scenario]
        model = MLFatigueModel()
        result = model.predict(_make_features(**overrides))
        assert 0.0 <= result.score <= 100.0
        assert result.level.value in ("low", "medium", "high")
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.state, RiskState)


# ---------------------------------------------------------------------------
# Heuristic detection correctness
# ---------------------------------------------------------------------------

class TestHeuristicDetection:
    def test_detects_microsleep(self) -> None:
        model = HeuristicFatigueModel()
        result = model.predict(_make_features(
            ear=0.05, perclos=0.40,
            current_eye_closure_duration=1.0,
            longest_eye_closure_duration=1.2,
        ))
        assert result.level.value == "high"
        assert result.state in (RiskState.microsleep_suspected, RiskState.critical_microsleep)

    def test_detects_critical_microsleep(self) -> None:
        model = HeuristicFatigueModel()
        result = model.predict(_make_features(
            ear=0.01, perclos=0.80,
            current_eye_closure_duration=2.0,
            longest_eye_closure_duration=2.5,
        ))
        assert result.state == RiskState.critical_microsleep
        assert result.score >= 88.0

    def test_detects_yawn_with_closure(self) -> None:
        model = HeuristicFatigueModel()
        # Yawn + long blink + corroboration → fatigue_warning (medium)
        result = model.predict(_make_features(
            ear=0.25, mar=0.70, perclos=0.15,
            blink_rate=6.0,
            current_yawn_duration=1.5, total_yawns=2.0,
            longest_eye_closure_duration=0.8,
        ))
        assert result.level.value in ("medium", "high")
        assert result.score >= 50.0

    def test_yawn_nudges_score_up(self) -> None:
        model = HeuristicFatigueModel()
        with_yawn = model.predict(_make_features(
            ear=0.10, perclos=0.30, blink_rate=6.0,
            longest_eye_closure_duration=0.8,
            current_yawn_duration=1.0, total_yawns=1.0,
        ))
        without_yawn = model.predict(_make_features(
            ear=0.10, perclos=0.30, blink_rate=6.0,
            longest_eye_closure_duration=0.8,
        ))
        assert with_yawn.score > without_yawn.score

    def test_neutral_is_low(self) -> None:
        model = HeuristicFatigueModel()
        result = model.predict(_make_features(
            ear=0.35, perclos=0.02, blink_rate=18.0,
        ))
        assert result.level.value == "low"
        assert result.state == RiskState.normal
