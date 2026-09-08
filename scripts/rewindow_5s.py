"""Re-window YAWDD and DDD frame-level data using 5-second windows (S3-05).

Reads existing frame-level parquets and generates new window-level parquets
with window_size=150 (5s at 30fps) and step_size=75 (50% overlap).

Usage:
    python scripts/rewindow_5s.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "dataset-manager"))

import pandas as pd

from dataset_manager.windowing import WindowingConfig, build_windows, windows_to_dataframe

PROCESSED_DIR = REPO_ROOT / "datasets" / "processed"

# 5-second window config
WINDOW_CONFIG_5S = WindowingConfig(
    window_size=150,        # 150 frames / 30fps = 5 seconds
    step_size=75,           # 50% overlap
    min_fill_ratio=0.5,     # at least 50% face-positive frames
    ear_closed_threshold=0.25,
    yawn_event_threshold=0.20,
    fps=30.0,
)

WINDOW_CONFIG_5S_DDD = WindowingConfig(
    window_size=150,        # 150 frames / 30fps = 5 seconds
    step_size=75,           # 50% overlap
    min_fill_ratio=0.3,     # lower threshold for DDD (independent images)
    ear_closed_threshold=0.25,
    yawn_event_threshold=0.20,
    fps=30.0,
)


def rewindow_yawdd() -> pd.DataFrame:
    """Re-window YAWDD frame-level data using 5s windows."""
    input_path = PROCESSED_DIR / "yawdd_features.parquet"
    output_path = PROCESSED_DIR / "yawdd_train_features.parquet"

    print(f"Reading YAWDD frame-level data from {input_path} ...")
    df = pd.read_parquet(input_path)
    print(f"  Frames: {len(df):,}")
    print(f"  Videos: {df['video_file'].nunique()}")

    print(f"\nWindowing config (5s):")
    print(f"  Window size: {WINDOW_CONFIG_5S.window_size} frames (~{WINDOW_CONFIG_5S.window_size / WINDOW_CONFIG_5S.fps:.1f}s)")
    print(f"  Step size:   {WINDOW_CONFIG_5S.step_size} frames (~{WINDOW_CONFIG_5S.step_size / WINDOW_CONFIG_5S.fps:.1f}s)")
    print(f"  Min fill:    {WINDOW_CONFIG_5S.min_fill_ratio * 100:.0f}%")

    t0 = time.time()
    records = build_windows(df, WINDOW_CONFIG_5S)
    elapsed = time.time() - t0
    print(f"\nBuilt {len(records):,} windows in {elapsed:.1f}s")

    if not records:
        print("ERROR: No windows generated.")
        sys.exit(1)

    result_df = windows_to_dataframe(records)
    result_df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported to {output_path} ({size_mb:.2f} MB)")
    print(f"Target distribution: {result_df['fatigue_state'].value_counts().to_dict()}")
    print(f"Subjects: {result_df['subject_id'].nunique()}")

    return result_df


def rewindow_ddd() -> pd.DataFrame:
    """Re-window DDD frame-level data using 5s windows."""
    input_path = PROCESSED_DIR / "ddd_features.parquet"
    output_path = PROCESSED_DIR / "ddd_train_features.parquet"

    print(f"\nReading DDD frame-level data from {input_path} ...")
    df = pd.read_parquet(input_path)
    print(f"  Frames: {len(df):,}")
    print(f"  Subjects: {df['subject_id'].nunique()}")

    # Ensure frame_id is sequential per subject
    df = df.copy()
    df["frame_id"] = df.groupby("subject_id").cumcount()

    print(f"\nWindowing config (5s, DDD):")
    print(f"  Window size: {WINDOW_CONFIG_5S_DDD.window_size} frames (~{WINDOW_CONFIG_5S_DDD.window_size / WINDOW_CONFIG_5S_DDD.fps:.1f}s)")
    print(f"  Step size:   {WINDOW_CONFIG_5S_DDD.step_size} frames (~{WINDOW_CONFIG_5S_DDD.step_size / WINDOW_CONFIG_5S_DDD.fps:.1f}s)")
    print(f"  Min fill:    {WINDOW_CONFIG_5S_DDD.min_fill_ratio * 100:.0f}%")

    t0 = time.time()
    records = build_windows(df, WINDOW_CONFIG_5S_DDD)
    elapsed = time.time() - t0
    print(f"\nBuilt {len(records):,} windows in {elapsed:.1f}s")

    if not records:
        print("ERROR: No DDD windows generated.")
        sys.exit(1)

    result_df = windows_to_dataframe(records)
    result_df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported to {output_path} ({size_mb:.2f} MB)")
    print(f"Target distribution: {result_df['fatigue_state'].value_counts().to_dict()}")
    print(f"Subjects: {result_df['subject_id'].nunique()}")

    return result_df


if __name__ == "__main__":
    yawdd_df = rewindow_yawdd()
    ddd_df = rewindow_ddd()

    print(f"\n{'='*60}")
    print(f"5s REWINDOWING COMPLETE")
    print(f"{'='*60}")
    print(f"YAWDD windows: {len(yawdd_df)}")
    print(f"DDD windows:   {len(ddd_df)}")
    print(f"Total:         {len(yawdd_df) + len(ddd_df)}")
