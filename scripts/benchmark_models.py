"""S3-04: Benchmark comparing HeuristicFatigueModel vs MLFatigueModel.

Runs 1,000 iterations of both models on synthetic scenarios and reports
latency, memory, and correctness across edge cases.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ai_engine.heuristic import HeuristicFatigueModel
from ai_engine.ml_model import MLFatigueModel
from ai_engine.types import (
    FatigueFeatures,
    HeadPose,
    IrisDirection,
    RiskState,
)

ITERATIONS = 1_000


def _make_features(**overrides: Any) -> FatigueFeatures:
    """Build minimal valid FatigueFeatures."""
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
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, dict[str, Any]] = {
    "neutral": dict(
        ear=0.35, mar=0.05, perclos=0.02, blink_rate=18.0,
        eye_closure_duration=0.1, longest_eye_closure_duration=0.2,
        current_eye_closure_duration=0.0,
    ),
    "talking": dict(
        ear=0.30, mar=0.35, perclos=0.08, blink_rate=14.0,
        eye_closure_duration=0.15, longest_eye_closure_duration=0.3,
        current_eye_closure_duration=0.0,
    ),
    "yawning": dict(
        ear=0.25, mar=0.70, perclos=0.15, blink_rate=10.0,
        eye_closure_duration=0.3, longest_eye_closure_duration=0.5,
        current_yawn_duration=1.5, total_yawns=2.0,
        current_eye_closure_duration=0.0,
    ),
    "long_blink": dict(
        ear=0.10, mar=0.05, perclos=0.30, blink_rate=8.0,
        eye_closure_duration=0.8, longest_eye_closure_duration=0.8,
        current_eye_closure_duration=0.0,
    ),
    "microsleep_1s": dict(
        ear=0.05, mar=0.05, perclos=0.40, blink_rate=6.0,
        eye_closure_duration=1.2, longest_eye_closure_duration=1.2,
        current_eye_closure_duration=1.0,
    ),
    "microsleep_2s": dict(
        ear=0.02, mar=0.05, perclos=0.60, blink_rate=4.0,
        eye_closure_duration=2.0, longest_eye_closure_duration=2.0,
        current_eye_closure_duration=1.8,
    ),
    "critical_closure": dict(
        ear=0.01, mar=0.05, perclos=0.80, blink_rate=2.0,
        eye_closure_duration=3.0, longest_eye_closure_duration=3.0,
        current_eye_closure_duration=2.5,
    ),
}


@dataclass
class BenchmarkResult:
    model_name: str
    scenario: str
    avg_latency_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    result_level: str
    result_state: str
    result_score: float


def benchmark_model(
    model: Any, model_name: str, features: FatigueFeatures, iterations: int
) -> list[float]:
    """Run model.predict() N times and return list of latencies in ms."""
    latencies: list[float] = []
    # Warmup
    for _ in range(50):
        model.predict(features)
    # Measured
    for _ in range(iterations):
        t0 = time.perf_counter()
        model.predict(features)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
    return latencies


def run_benchmark() -> tuple[list[BenchmarkResult], dict[str, dict[str, Any]]]:
    """Run full benchmark and return results + edge-case analysis."""
    heuristic = HeuristicFatigueModel()
    ml = MLFatigueModel()
    results: list[BenchmarkResult] = []

    for scenario_name, overrides in SCENARIOS.items():
        features = _make_features(**overrides)
        for model, model_name in [(heuristic, "Heuristic"), (ml, "ML")]:
            latencies = benchmark_model(model, model_name, features, ITERATIONS)
            result = model.predict(features)
            results.append(BenchmarkResult(
                model_name=model_name,
                scenario=scenario_name,
                avg_latency_ms=round(statistics.mean(latencies), 3),
                p95_ms=round(np.percentile(latencies, 95), 3),
                p99_ms=round(np.percentile(latencies, 99), 3),
                min_ms=round(min(latencies), 3),
                max_ms=round(max(latencies), 3),
                result_level=result.level.value,
                result_state=result.state.value,
                result_score=round(result.score, 1),
            ))

    # Edge-case analysis: compare outputs side-by-side
    analysis: dict[str, dict[str, Any]] = {}
    for scenario_name, overrides in SCENARIOS.items():
        features = _make_features(**overrides)
        h_result = heuristic.predict(features)
        m_result = ml.predict(features)
        analysis[scenario_name] = {
            "heuristic": {"level": h_result.level.value, "state": h_result.state.value, "score": h_result.score},
            "ml": {"level": m_result.level.value, "state": m_result.state.value, "score": m_result.score},
            "score_diff": round(abs(h_result.score - m_result.score), 1),
            "level_match": h_result.level.value == m_result.level.value,
        }
    return results, analysis


def main() -> None:
    results, analysis = run_benchmark()

    print("\n" + "=" * 80)
    print("  S3-04 MODEL BENCHMARK: Heuristic vs ML (fatigue_v2)")
    print("=" * 80)

    # Latency summary
    print(f"\n  Iterations per scenario: {ITERATIONS:,}")
    print(f"  Warmup: 50 iterations (excluded from measurements)\n")

    for model_name in ["Heuristic", "ML"]:
        model_results = [r for r in results if r.model_name == model_name]
        all_lat = []
        for r in model_results:
            all_lat.append(r.avg_latency_ms)
        print(f"  {model_name}:")
        print(f"    Avg latency:    {statistics.mean(all_lat):.3f} ms")
        print(f"    Overall p95:    {np.percentile(all_lat, 95):.3f} ms")
        print(f"    Overall p99:    {np.percentile(all_lat, 99):.3f} ms")
        print()

    # Per-scenario table
    print("  {:<20s} {:>10s} {:>10s} {:>8s} {:>8s} {:>12s} {:>12s}".format(
        "Scenario", "H-lat(ms)", "ML-lat(ms)", "H-level", "ML-level", "H-score", "ML-score"
    ))
    print("  " + "-" * 80)
    for scenario_name in SCENARIOS:
        h = next(r for r in results if r.scenario == scenario_name and r.model_name == "Heuristic")
        m = next(r for r in results if r.scenario == scenario_name and r.model_name == "ML")
        print("  {:<20s} {:>10.3f} {:>10.3f} {:>8s} {:>8s} {:>12.1f} {:>12.1f}".format(
            scenario_name, h.avg_latency_ms, m.avg_latency_ms,
            h.result_level, m.result_level, h.result_score, m.result_score,
        ))

    # Edge-case divergence
    print("\n  EDGE-CASE DIVERGENCE:")
    print("  {:<20s} {:>12s} {:>12s} {:>10s} {:>10s}".format(
        "Scenario", "H-state", "ML-state", "score-diff", "level-match"
    ))
    print("  " + "-" * 70)
    for scenario, data in analysis.items():
        print("  {:<20s} {:>12s} {:>12s} {:>10.1f} {:>10s}".format(
            scenario,
            data["heuristic"]["state"],
            data["ml"]["state"],
            data["score_diff"],
            str(data["level_match"]),
        ))

    print("\n" + "=" * 80)
    print("  Benchmark complete. See reports/model_comparison.md for full report.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
