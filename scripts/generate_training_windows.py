"""Generate ML training windows from frame-level features.

Usage::

    python scripts/generate_training_windows.py

Reads ``datasets/processed/yawdd_features.parquet`` (frame-level),
applies sliding-window aggregation, and writes
``datasets/processed/yawdd_train_features.parquet`` (window-level).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "dataset-manager"))

import pandas as pd

from dataset_manager.windowing import WindowingConfig, build_windows, windows_to_dataframe


def main() -> None:
    input_path = Path("datasets/processed/yawdd_features.parquet")
    output_path = Path("datasets/processed/yawdd_train_features.parquet")

    print(f"Reading frame-level data from {input_path} ...")
    df = pd.read_parquet(input_path)
    print(f"  Frames: {len(df):,}")
    print(f"  Videos: {df['video_file'].nunique()}")

    # Configure windowing: 5-second windows (150 frames @ 30 FPS), 50% overlap
    config = WindowingConfig(
        window_size=150,      # ~5 seconds at 30 FPS
        step_size=75,         # 50% overlap
        min_fill_ratio=0.5,   # at least 50% face-positive frames
        ear_closed_threshold=0.25,
        yawn_event_threshold=0.20,
        fps=30.0,
    )
    print(f"\nWindowing config:")
    print(f"  Window size: {config.window_size} frames (~{config.window_size / config.fps:.1f}s)")
    print(f"  Step size:   {config.step_size} frames (~{config.step_size / config.fps:.1f}s)")
    print(f"  Min fill:    {config.min_fill_ratio * 100:.0f}%")
    print(f"  EAR thresh:  {config.ear_closed_threshold}")
    print(f"  Yawn thresh: {config.yawn_event_threshold * 100:.0f}%")

    t0 = time.time()
    records = build_windows(df, config)
    elapsed = time.time() - t0
    print(f"\nBuilt {len(records):,} windows in {elapsed:.1f}s")

    if not records:
        print("ERROR: No windows generated. Check input data and config.")
        sys.exit(1)

    # Convert to DataFrame and export
    train_df = windows_to_dataframe(records)
    train_df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported to {output_path} ({size_mb:.2f} MB)")

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"TRAINING DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total windows:   {len(train_df):,}")
    print(f"Total videos:    {train_df['video_file'].nunique()}")
    print(f"Avg frames/win:  {train_df['frame_count'].mean():.0f}")
    print(f"Avg face/win:    {train_df['face_frame_count'].mean():.0f}")

    print(f"\n--- Target distribution (fatigue_state) ---")
    target_counts = train_df["fatigue_state"].value_counts()
    for label, count in target_counts.items():
        pct = count / len(train_df) * 100
        print(f"  {label:<20s} {count:>6,}  ({pct:5.1f}%)")

    print(f"\n--- Gender distribution ---")
    for g, count in train_df["gender"].value_counts().items():
        pct = count / len(train_df) * 100
        print(f"  {g:<20s} {count:>6,}  ({pct:5.1f}%)")

    print(f"\n--- Glasses distribution ---")
    for g, count in train_df["glasses"].value_counts().items():
        pct = count / len(train_df) * 100
        print(f"  {g:<20s} {count:>6,}  ({pct:5.1f}%)")

    print(f"\n--- Camera set distribution ---")
    for c, count in train_df["camera_set"].value_counts().items():
        pct = count / len(train_df) * 100
        print(f"  {c:<20s} {count:>6,}  ({pct:5.1f}%)")

    print(f"\n--- Feature ranges (across windows) ---")
    print(f"  EAR mean:  [{train_df['ear_mean'].min():.4f}, {train_df['ear_mean'].max():.4f}]")
    print(f"  MAR mean:  [{train_df['mar_mean'].min():.4f}, {train_df['mar_mean'].max():.4f}]")
    print(f"  PERCLOS:   [{train_df['perclos'].min():.4f}, {train_df['perclos'].max():.4f}]")
    print(f"  Blink rt:  [{train_df['blink_rate'].min():.1f}, {train_df['blink_rate'].max():.1f}] blinks/min")
    print(f"  Yawn frac: [{train_df['yawn_fraction'].min():.4f}, {train_df['yawn_fraction'].max():.4f}]")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
