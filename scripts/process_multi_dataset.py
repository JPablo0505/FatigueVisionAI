#!/usr/bin/env python3
"""Multi-dataset processing pipeline: YAWDD + DDD → fused training data.

This script:
1. Processes DDD images through the feature extractor (frame-level).
2. Generates temporal windows for both YAWDD and DDD.
3. Fuses the windowed data into unified train/val/test parquets.
4. Runs GroupKFold CV and promotes fatigue_v2.joblib.

Usage::

    python scripts/process_multi_dataset.py

Output:
    datasets/processed/ddd_features.parquet        (DDD frame-level)
    datasets/processed/ddd_train_features.parquet   (DDD window-level)
    datasets/processed/fused_train_features.parquet (merged windows)
    datasets/processed/fused_train.parquet          (split: train)
    datasets/processed/fused_val.parquet            (split: val)
    datasets/processed/fused_test.parquet           (split: test)
    packages/ai-engine/ai_engine/models/fatigue_v2.joblib
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure the repo root is on sys.path for local imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "dataset-manager"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "feature-extractor"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "ai-engine"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "vision-engine"))

import numpy as np
import pandas as pd

from dataset_manager.adapters.ddd import DDDAdapter, DDDConfig, _list_image_files
from dataset_manager.preprocessing import (
    CleaningConfig,
    compute_cleaning_stats,
    samples_to_dataframe,
)

# Frame-level scalar features to extract from FatigueFeatures
# (nested head_pose/iris handled separately)
_DDD_FEATURE_FIELDS = [
    "ear", "mar", "perclos", "blink_rate",
    "eye_closure_duration", "yawn_duration",
]
from dataset_manager.processor import DatasetProcessor, ProcessedSample
from dataset_manager.windowing import WindowingConfig, build_windows, windows_to_dataframe
from dataset_manager.splitting import SplitConfig, grouped_stratified_split, split_summary, validate_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("process_multi_dataset")


# ---------------------------------------------------------------------------
# DDD frame-level processing
# ---------------------------------------------------------------------------

class ResilientProcessor(DatasetProcessor):
    """DatasetProcessor that catches InvalidFeaturesError per-frame."""

    def process(self, adapter, *, max_samples=None):
        """Override to catch per-frame feature extraction errors."""
        info = adapter.info()
        logger.info(
            "Processing dataset '%s' v%s (%d subjects, %d frames)",
            info.name, info.version, info.subjects, info.total_frames,
        )

        count = 0
        last_clip = None
        error_count = 0

        for event, labels in adapter.iter_events():
            # Reset window at clip boundaries
            clip_id = str(labels.get("clip_id", ""))
            if self._reset_between_clips and clip_id != last_clip and last_clip is not None:
                self._extractor.reset()
            last_clip = clip_id

            # Extract features with error handling
            try:
                features = self._extractor.update(event)
                has_face = features is not None
            except Exception as e:
                logger.debug(
                    "Frame %d: feature extraction error: %s", event.frame_id, e
                )
                features = None
                has_face = False
                error_count += 1

            yield ProcessedSample(
                features=features,
                labels=labels,
                frame_id=event.frame_id,
                has_face=has_face,
            )

            count += 1
            if max_samples is not None and count >= max_samples:
                logger.info("Reached max_samples limit (%d)", max_samples)
                break

        if error_count > 0:
            logger.warning(
                "Feature extraction errors: %d frames skipped (%.1f%% of total)",
                error_count,
                error_count / max(count, 1) * 100,
            )
        logger.info("Processed %d samples from dataset '%s'", count, info.name)


def process_ddd(
    ddd_root: Path,
    output_path: Path,
    model_dir: Path,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Process DDD dataset and save frame-level features to Parquet."""
    logger.info("=" * 60)
    logger.info("DDD Dataset Processing")
    logger.info("=" * 60)
    logger.info("Input:  %s", ddd_root)
    logger.info("Output: %s", output_path)

    t_start = time.time()

    # Configure adapter
    config = DDDConfig(
        ddd_root=ddd_root,
        face_detector=model_dir / "face_detector.tflite",
        face_landmarker=model_dir / "face_landmarker.task",
    )
    adapter = DDDAdapter(config)

    # Validate
    logger.info("Validating dataset...")
    errors = adapter.validate()
    if errors:
        for e in errors:
            logger.error("  VALIDATION: %s", e)
        sys.exit(1)
    logger.info("Validation passed.")

    # Process through FeatureExtractor
    logger.info("Starting feature extraction...")
    processor = ResilientProcessor()

    try:
        from tqdm import tqdm

        samples_iter = processor.process(adapter, max_samples=max_samples)
        pbar = tqdm(desc="Processing DDD frames", unit="frame", dynamic_ncols=True)
        processed_samples = []
        for sample in samples_iter:
            processed_samples.append(sample)
            pbar.update(1)
            if len(processed_samples) % 500 == 0:
                pbar.set_postfix(
                    faces=sum(1 for s in processed_samples if s.has_face),
                    total=len(processed_samples),
                )
        pbar.close()

    except ImportError:
        logger.warning("tqdm not available, processing without progress bar")
        processed_samples = list(processor.process(adapter, max_samples=max_samples))

    t_extract = time.time()
    logger.info(
        "Feature extraction complete: %d samples in %.1fs",
        len(processed_samples), t_extract - t_start,
    )

    if not processed_samples:
        logger.error("No samples produced. Check DDD data and model files.")
        sys.exit(1)

    # Build DataFrame manually — samples_to_dataframe drops DDD-specific labels
    logger.info("Building DataFrame with full label preservation...")
    rows = []
    for sample in processed_samples:
        row = {}
        # Feature fields
        feat = sample.features
        if feat is not None:
            for f in _DDD_FEATURE_FIELDS:
                row[f] = getattr(feat, f, None)
            if feat.head_pose is not None:
                row["head_pose_yaw"] = feat.head_pose.yaw
                row["head_pose_pitch"] = feat.head_pose.pitch
                row["head_pose_roll"] = feat.head_pose.roll
            if feat.iris_direction is not None:
                row["iris_x"] = feat.iris_direction.x
                row["iris_y"] = feat.iris_direction.y
        # DDD-specific labels (preserved from adapter)
        row["condition"] = sample.labels.get("fatigue_level", None)  # drowsy/non_drowsy
        row["class_dir"] = sample.labels.get("class_dir", None)      # drowsy/non_drowsy
        row["subject_id"] = sample.labels.get("subject_id", None)
        row["clip_id"] = sample.labels.get("clip_id", None)
        row["frame_id"] = sample.frame_id
        row["has_face"] = sample.has_face
        rows.append(row)

    df = pd.DataFrame(rows)

    # Map DDD class_dir to YAWDD condition format for target labeling
    # drowsy → yawning (high fatigue), non_drowsy → normal (low fatigue)
    if "class_dir" in df.columns:
        df["condition"] = df["class_dir"].map({
            "drowsy": "yawning",
            "non_drowsy": "normal",
        })
        logger.info("Mapped class_dir → condition: %s", df["condition"].value_counts().to_dict())

    # Fill missing columns expected by windowing engine
    df["video_file"] = df.get("video_file", pd.Series([None] * len(df))).fillna(df["subject_id"])
    df["gender"] = df.get("gender", pd.Series([None] * len(df))).fillna("unknown")
    df["glasses"] = df.get("glasses", pd.Series([None] * len(df))).fillna("unknown")
    df["camera_set"] = df.get("camera_set", pd.Series([None] * len(df))).fillna("ddd")

    # Compute stats
    total_frames = len(df)
    faces_with = df["has_face"].sum()
    face_rate = (faces_with / total_frames * 100) if total_frames > 0 else 0

    # Save to Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")

    t_save = time.time()
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info("=" * 60)
    logger.info("DDD PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info("Total frames: %d", total_frames)
    logger.info("Frames with face: %d (%.1f%%)", faces_with, face_rate)
    logger.info("Output: %s (%.1f MB)", output_path, file_size_mb)
    logger.info("Total time: %.1fs", time.time() - t_start)

    return df


# ---------------------------------------------------------------------------
# DDD windowing (group by subject_id, not video_file)
# ---------------------------------------------------------------------------

def window_ddd_by_subject(
    df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Generate windows from DDD frame-level data, grouped by subject_id.

    DDD images are independent (no temporal continuity), so we treat each
    subject's images as a single "virtual video" and create one aggregated
    window per subject.
    """
    logger.info("=" * 60)
    logger.info("DDD Windowing (by subject_id)")
    logger.info("=" * 60)
    logger.info("Input frames: %d", len(df))
    logger.info("Subjects: %d", df["subject_id"].nunique())

    df = df.copy()

    # Ensure frame_id is sequential per subject (required by windowing engine)
    df["frame_id"] = df.groupby("subject_id").cumcount()

    # Use a window that captures subject-level patterns
    # DDD subjects have varying image counts (400-3600)
    config = WindowingConfig(
        window_size=150,       # ~5 seconds equivalent
        step_size=75,          # 50% overlap
        min_fill_ratio=0.3,    # Lower threshold since images are independent
        ear_closed_threshold=0.25,
        yawn_event_threshold=0.20,
        fps=30.0,
    )

    logger.info("Window config: size=%d, step=%d, min_fill=%.0f%%",
                config.window_size, config.step_size, config.min_fill_ratio * 100)

    t0 = time.time()
    records = build_windows(df, config)
    elapsed = time.time() - t0
    logger.info("Built %d windows in %.1fs", len(records), elapsed)

    if not records:
        logger.warning("No windows generated. Trying with smaller window size...")
        config = WindowingConfig(
            window_size=100,
            step_size=100,
            min_fill_ratio=0.1,
        )
        records = build_windows(df, config)
        logger.info("Built %d windows with smaller window", len(records))

    if not records:
        logger.error("ERROR: No DDD windows generated.")
        sys.exit(1)

    # Convert to DataFrame and export
    train_df = windows_to_dataframe(records)
    train_df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info("Exported %d windows to %s (%.2f MB)", len(train_df), output_path, size_mb)
    logger.info("Target distribution: %s", train_df["fatigue_state"].value_counts().to_dict())
    logger.info("Subjects: %d", train_df["subject_id"].nunique())

    return train_df


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def fuse_datasets(
    yawdd_windows_path: Path,
    ddd_windows_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Merge YAWDD and DDD window-level data into a unified dataset."""
    logger.info("=" * 60)
    logger.info("Dataset Fusion: YAWDD + DDD")
    logger.info("=" * 60)

    yawdd_df = pd.read_parquet(yawdd_windows_path)
    ddd_df = pd.read_parquet(ddd_windows_path)

    logger.info("YAWDD windows: %d (subjects: %d)", len(yawdd_df), yawdd_df["subject_id"].nunique())
    logger.info("DDD windows:   %d (subjects: %d)", len(ddd_df), ddd_df["subject_id"].nunique())

    # Ensure common columns exist
    for col in ["video_file", "gender", "glasses", "camera_set"]:
        if col not in ddd_df.columns:
            ddd_df[col] = "unknown" if col != "video_file" else ddd_df.get("subject_id", "unknown")

    # Select common columns (intersection)
    common_cols = list(set(yawdd_df.columns) & set(ddd_df.columns))
    logger.info("Common columns: %d", len(common_cols))

    # Fuse
    fused_df = pd.concat([yawdd_df[common_cols], ddd_df[common_cols]], ignore_index=True)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fused_df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info("Fused dataset: %d windows, %d subjects", len(fused_df), fused_df["subject_id"].nunique())
    logger.info("Target distribution: %s", fused_df["fatigue_state"].value_counts().to_dict())
    logger.info("Saved to %s (%.2f MB)", output_path, size_mb)

    return fused_df


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_fused_dataset(fused_df: pd.DataFrame, output_dir: Path) -> None:
    """Apply grouped stratified split to the fused dataset."""
    logger.info("=" * 60)
    logger.info("Splitting Fused Dataset")
    logger.info("=" * 60)

    config = SplitConfig(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        group_col="subject_id",
        target_col="fatigue_state",
    )

    logger.info("Split: 70/15/15 by subject_id")
    result = grouped_stratified_split(fused_df, config)

    # Validate
    errors = validate_split(result)
    if errors:
        logger.error("VALIDATION ERRORS:")
        for e in errors:
            logger.error("  - %s", e)
        sys.exit(1)
    logger.info("Validation passed: zero data leakage")

    # Save
    train_path = output_dir / "fused_train.parquet"
    val_path = output_dir / "fused_val.parquet"
    test_path = output_dir / "fused_test.parquet"

    result.train.to_parquet(train_path, index=False)
    result.val.to_parquet(val_path, index=False)
    result.test.to_parquet(test_path, index=False)

    logger.info("Saved:")
    logger.info("  %s (%d windows)", train_path, len(result.train))
    logger.info("  %s (%d windows)", val_path, len(result.val))
    logger.info("  %s (%d windows)", test_path, len(result.test))

    # Summary
    summary = split_summary(result)
    logger.info("Total windows: %d, Subjects: %d", summary["total_windows"], summary["total_subjects"])

    for name in ["train", "val", "test"]:
        s = summary["splits"][name]
        logger.info("  %s: %d windows (%s%%), %d subjects, targets: %s",
                     name.upper(), s["windows"], s["pct"], s["subjects"], s["target_counts"])

    # Save summary
    summary_path = output_dir / "fused_split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-dataset processing pipeline (YAWDD + DDD → fatigue_v2)"
    )
    parser.add_argument(
        "--ddd-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "raw" / "Driver Drowsiness Dataset (DDD)",
        help="Root directory of the DDD dataset.",
    )
    parser.add_argument(
        "--yawdd-windows",
        type=Path,
        default=REPO_ROOT / "datasets" / "processed" / "yawdd_train_features.parquet",
        help="Pre-existing YAWDD window-level parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "processed",
        help="Output directory for processed data.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "models" / "vision",
        help="Directory containing MediaPipe model files.",
    )
    parser.add_argument(
        "--max-ddd-samples",
        type=int,
        default=None,
        help="Max DDD images to process (None = all).",
    )
    parser.add_argument(
        "--skip-ddd-processing",
        action="store_true",
        help="Skip DDD frame-level processing (use existing parquet).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    t_total_start = time.time()

    # --- 1. Process DDD frame-level ---
    ddd_features_path = output_dir / "ddd_features.parquet"
    ddd_windows_path = output_dir / "ddd_train_features.parquet"

    if not args.skip_ddd_processing:
        process_ddd(
            ddd_root=args.ddd_root,
            output_path=ddd_features_path,
            model_dir=args.model_dir,
            max_samples=args.max_ddd_samples,
        )
    else:
        logger.info("Skipping DDD processing, using existing %s", ddd_features_path)

    # --- 2. Window DDD by subject ---
    if not ddd_windows_path.exists() or not args.skip_ddd_processing:
        ddd_df = pd.read_parquet(ddd_features_path)
        window_ddd_by_subject(ddd_df, ddd_windows_path)
    else:
        logger.info("Using existing DDD windows: %s", ddd_windows_path)

    # --- 3. Fuse YAWDD + DDD ---
    fused_path = output_dir / "fused_train_features.parquet"
    fused_df = fuse_datasets(args.yawdd_windows, ddd_windows_path, fused_path)

    # --- 4. Split ---
    split_fused_dataset(fused_df, output_dir)

    t_total = time.time() - t_total_start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.1fs", t_total)
    logger.info("=" * 60)
    logger.info("Next: run train_pipeline.py with fused data paths")


if __name__ == "__main__":
    main()
