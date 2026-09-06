"""Reproducible training pipeline with GroupKFold cross-validation (S2-06).

Usage::

    # Default config
    python scripts/train_pipeline.py

    # Custom config
    python scripts/train_pipeline.py --config config/train_config.yaml

    # Override individual settings
    python scripts/train_pipeline.py --strategy holdout --n-estimators 200

    # Reproduce exact run
    python scripts/train_pipeline.py --config experiments/<timestamp>/config.json

Each run creates an experiment directory under ``experiments/<timestamp>/``
containing config.json, metrics.json, and model.joblib.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------------------
# Defaults (used when no config is provided)
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    "data": {
        "train_path": "datasets/processed/yawdd_train.parquet",
        "val_path": "datasets/processed/yawdd_val.parquet",
        "test_path": "datasets/processed/yawdd_test.parquet",
    },
    "features": {
        "columns": [
            "ear_mean", "ear_std", "ear_min", "ear_max",
            "mar_mean", "mar_std", "mar_min", "mar_max",
            "perclos", "blink_rate",
            "eye_closure_mean", "eye_closure_max",
            "yawn_mean", "yawn_max", "yawn_fraction",
            "head_pose_yaw_mean", "head_pose_pitch_mean", "head_pose_roll_mean",
            "iris_x_mean", "iris_y_mean",
        ],
        "target_map": {
            "normal": "low", "talking": "low",
            "unknown": "medium", "yawning": "high",
        },
        "group_column": "subject_id",
    },
    "model": {
        "classifier": "RandomForestClassifier",
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
        },
        "random_state": 42,
    },
    "evaluation": {
        "strategy": "group_kfold",
        "group_kfold": {"n_splits": 5},
        "eval_holdout_val": True,
        "eval_holdout_test": True,
    },
    "experiments": {
        "base_dir": "experiments",
        "save_model": True,
        "save_config": True,
        "save_metrics": True,
    },
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file, falling back to defaults."""
    config = deepcopy(DEFAULTS)
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                loaded = yaml.safe_load(f)
            if loaded:
                _deep_merge(config, loaded)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments that override config values."""
    parser = argparse.ArgumentParser(
        description="FatigueVisionAI Training Pipeline (S2-06)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--train-path", type=str, default=None, help="Training data path")
    parser.add_argument("--val-path", type=str, default=None, help="Validation data path")
    parser.add_argument("--test-path", type=str, default=None, help="Test data path")
    parser.add_argument("--strategy", choices=["group_kfold", "holdout"], default=None,
                        help="Evaluation strategy")
    parser.add_argument("--n-splits", type=int, default=None, help="Number of CV folds")
    parser.add_argument("--n-estimators", type=int, default=None, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=None, help="Max tree depth")
    parser.add_argument("--random-state", type=int, default=None, help="Random seed")
    parser.add_argument("--experiments-dir", type=str, default=None, help="Experiments root dir")
    parser.add_argument("--no-holdout-val", action="store_true", help="Skip holdout val eval")
    parser.add_argument("--no-holdout-test", action="store_true", help="Skip holdout test eval")
    parser.add_argument("--quiet", action="store_true", help="Suppress printed output")
    return parser.parse_args(argv)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply CLI overrides to the loaded config."""
    if args.train_path:
        config["data"]["train_path"] = args.train_path
    if args.val_path:
        config["data"]["val_path"] = args.val_path
    if args.test_path:
        config["data"]["test_path"] = args.test_path
    if args.strategy:
        config["evaluation"]["strategy"] = args.strategy
    if args.n_splits:
        config["evaluation"]["group_kfold"]["n_splits"] = args.n_splits
    if args.n_estimators:
        config["model"]["params"]["n_estimators"] = args.n_estimators
    if args.max_depth:
        config["model"]["params"]["max_depth"] = args.max_depth
    if args.random_state is not None:
        config["model"]["random_state"] = args.random_state
    if args.experiments_dir:
        config["experiments"]["base_dir"] = args.experiments_dir
    if args.no_holdout_val:
        config["evaluation"]["eval_holdout_val"] = False
    if args.no_holdout_test:
        config["evaluation"]["eval_holdout_test"] = False
    return config


# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------

def create_experiment_dir(base_dir: str | Path) -> tuple[Path, str]:
    """Create a timestamped experiment directory. Returns (path, timestamp_str)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    exp_dir = Path(base_dir) / ts
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir, ts


def save_experiment_config(exp_dir: Path, config: dict[str, Any]) -> Path:
    """Save the full config as JSON in the experiment directory."""
    path = exp_dir / "config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    return path


def save_experiment_metrics(exp_dir: Path, metrics: dict[str, Any]) -> Path:
    """Save metrics as JSON in the experiment directory."""
    path = exp_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return path


def save_experiment_model(
    exp_dir: Path, artifact: dict[str, Any], symlink_name: str = "model.joblib"
) -> Path:
    """Save the model artifact as joblib in the experiment directory."""
    path = exp_dir / symlink_name
    joblib.dump(artifact, path)
    return path


def create_latest_symlink(base_dir: Path, exp_dir: Path) -> Path | None:
    """Create or update a 'latest' symlink pointing to the newest experiment."""
    latest = base_dir / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        # Use relative path for portability
        latest.symlink_to(exp_dir.relative_to(base_dir))
        return latest
    except OSError:
        # Windows without admin privileges may not support symlinks
        return None


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute a standard metric set from true/predicted labels."""
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def train_model(
    X: np.ndarray, y: np.ndarray, config: dict[str, Any]
) -> RandomForestClassifier:
    """Train a RandomForestClassifier from config."""
    params = deepcopy(config["model"]["params"])
    params["random_state"] = config["model"]["random_state"]
    clf = RandomForestClassifier(**params)
    clf.fit(X, y)
    return clf


def run_group_kfold_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_map: dict[str, str],
    group_col: str,
    config: dict[str, Any],
    *,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run GroupKFold cross-validation and return aggregated metrics."""
    n_splits = config["evaluation"]["group_kfold"]["n_splits"]
    random_state = config["model"]["random_state"]

    X = df[feature_cols].values
    y_raw = df["fatigue_state"].values
    groups = df[group_col].values
    y = np.array([target_map.get(c, "medium") for c in y_raw])

    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics: list[dict[str, float]] = []
    fold_details: list[dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = train_model(X_train, y_train, config)
        y_pred = clf.predict(X_val)

        metrics = compute_metrics(y_val, y_pred)
        fold_metrics.append(metrics)

        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        fold_detail = {
            "fold": fold_idx + 1,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "train_subjects": len(train_groups),
            "val_subjects": len(val_groups),
            "val_subject_ids": sorted(val_groups),
            "metrics": metrics,
        }
        fold_details.append(fold_detail)

        if not quiet:
            print(f"  Fold {fold_idx + 1}/{n_splits}: "
                  f"train={len(train_idx)} ({len(train_groups)} subjects), "
                  f"val={len(val_idx)} ({len(val_groups)} subjects), "
                  f"F1-macro={metrics['f1_macro']:.4f}, "
                  f"F1-weighted={metrics['f1_weighted']:.4f}")

    # Aggregate: mean and std across folds
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    aggregated = {}
    for key in metric_keys:
        values = [m[key] for m in fold_metrics]
        aggregated[f"{key}_mean"] = round(float(np.mean(values)), 4)
        aggregated[f"{key}_std"] = round(float(np.std(values)), 4)

    return {
        "strategy": "group_kfold",
        "n_splits": n_splits,
        "fold_details": fold_details,
        "aggregated": aggregated,
    }


def run_holdout_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict[str, Any],
    *,
    label: str = "holdout_val",
) -> dict[str, Any]:
    """Train on train set, evaluate on holdout set."""
    clf = train_model(X_train, y_train, config)
    y_pred = clf.predict(X_val)
    metrics = compute_metrics(y_val, y_pred)
    return {"set": label, "samples": len(y_val), "metrics": metrics}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(argv: list[str] | None = None) -> dict[str, Any]:
    """Execute the full training pipeline. Returns the final metrics dict."""
    args = parse_args(argv)
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    quiet = args.quiet

    # -- Unpack config --
    feature_cols: list[str] = config["features"]["columns"]
    target_map: dict[str, str] = config["features"]["target_map"]
    group_col: str = config["features"]["group_column"]

    # -- Load data --
    if not quiet:
        print("=" * 60)
        print("FatigueVisionAI Training Pipeline (S2-06)")
        print("=" * 60)

    train_path = Path(config["data"]["train_path"])
    val_path = Path(config["data"]["val_path"])
    test_path = Path(config["data"]["test_path"])

    if not quiet:
        print(f"\nLoading training data from {train_path} ...")
    train_df = pd.read_parquet(train_path)
    if not quiet:
        print(f"  Windows: {len(train_df)}, Subjects: {train_df[group_col].nunique()}")

    val_df = None
    if val_path.exists():
        if not quiet:
            print(f"Loading validation data from {val_path} ...")
        val_df = pd.read_parquet(val_path)
        if not quiet:
            print(f"  Windows: {len(val_df)}, Subjects: {val_df[group_col].nunique()}")

    test_df = None
    if test_path.exists():
        if not quiet:
            print(f"Loading test data from {test_path} ...")
        test_df = pd.read_parquet(test_path)
        if not quiet:
            print(f"  Windows: {len(test_df)}, Subjects: {test_df[group_col].nunique()}")

    # -- Prepare labels --
    y_train_raw = train_df["fatigue_state"].values
    y_train = np.array([target_map.get(c, "medium") for c in y_train_raw])
    if not quiet:
        print(f"\nTarget mapping: {target_map}")
        print(f"Train labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    strategy = config["evaluation"]["strategy"]
    all_metrics: dict[str, Any] = {
        "config": {
            "strategy": strategy,
            "model": config["model"],
            "features_count": len(feature_cols),
            "train_windows": len(train_df),
            "train_subjects": int(train_df[group_col].nunique()),
        }
    }

    # -- Cross-validation --
    if strategy == "group_kfold":
        if not quiet:
            n_splits = config["evaluation"]["group_kfold"]["n_splits"]
            print(f"\n{'='*60}")
            print(f"GROUPKFOLD CROSS-VALIDATION (K={n_splits})")
            print(f"{'='*60}")
        cv_result = run_group_kfold_cv(
            train_df, feature_cols, target_map, group_col, config, quiet=quiet
        )
        all_metrics["cross_validation"] = cv_result
        if not quiet:
            agg = cv_result["aggregated"]
            print(f"\nAggregated CV Metrics (mean ± std):")
            print(f"  Accuracy:      {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}")
            print(f"  Precision:     {agg['precision_macro_mean']:.4f} ± {agg['precision_macro_std']:.4f}")
            print(f"  Recall:        {agg['recall_macro_mean']:.4f} ± {agg['recall_macro_std']:.4f}")
            print(f"  F1 (macro):    {agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}")
            print(f"  F1 (weighted): {agg['f1_weighted_mean']:.4f} ± {agg['f1_weighted_std']:.4f}")

    # -- Holdout validation eval --
    if config["evaluation"].get("eval_holdout_val") and val_df is not None:
        if not quiet:
            print(f"\n{'='*60}")
            print(f"HOLDOUT VALIDATION EVALUATION")
            print(f"{'='*60}")
        X_val = val_df[feature_cols].values
        y_val_raw = val_df["fatigue_state"].values
        y_val = np.array([target_map.get(c, "medium") for c in y_val_raw])
        holdout_result = run_holdout_eval(
            train_df[feature_cols].values, y_train, X_val, y_val, config, label="holdout_val"
        )
        all_metrics["holdout_val"] = holdout_result
        if not quiet:
            m = holdout_result["metrics"]
            print(f"  Samples: {holdout_result['samples']}")
            print(f"  Accuracy:      {m['accuracy']:.4f}")
            print(f"  Precision:     {m['precision_macro']:.4f}")
            print(f"  Recall:        {m['recall_macro']:.4f}")
            print(f"  F1 (macro):    {m['f1_macro']:.4f}")
            print(f"  F1 (weighted): {m['f1_weighted']:.4f}")

    # -- Holdout test eval --
    if config["evaluation"].get("eval_holdout_test") and test_df is not None:
        if not quiet:
            print(f"\n{'='*60}")
            print(f"HOLDOUT TEST EVALUATION")
            print(f"{'='*60}")
        X_test = test_df[feature_cols].values
        y_test_raw = test_df["fatigue_state"].values
        y_test = np.array([target_map.get(c, "medium") for c in y_test_raw])
        holdout_result = run_holdout_eval(
            train_df[feature_cols].values, y_train, X_test, y_test, config, label="holdout_test"
        )
        all_metrics["holdout_test"] = holdout_result
        if not quiet:
            m = holdout_result["metrics"]
            print(f"  Samples: {holdout_result['samples']}")
            print(f"  Accuracy:      {m['accuracy']:.4f}")
            print(f"  Precision:     {m['precision_macro']:.4f}")
            print(f"  Recall:        {m['recall_macro']:.4f}")
            print(f"  F1 (macro):    {m['f1_macro']:.4f}")
            print(f"  F1 (weighted): {m['f1_weighted']:.4f}")

    # -- Final model training on full train set --
    if not quiet:
        print(f"\n{'='*60}")
        print(f"TRAINING FINAL MODEL ON FULL TRAIN SET")
        print(f"{'='*60}")
    t0 = time.time()
    final_model = train_model(train_df[feature_cols].values, y_train, config)
    train_time = time.time() - t0
    all_metrics["final_model"] = {
        "train_time_seconds": round(train_time, 2),
        "train_samples": len(train_df),
        "feature_importances": {
            name: round(float(imp), 4)
            for name, imp in zip(feature_cols, final_model.feature_importances_)
        },
    }
    if not quiet:
        print(f"  Training completed in {train_time:.2f}s")
        importances = all_metrics["final_model"]["feature_importances"]
        top5 = sorted(importances.items(), key=lambda x: -x[1])[:5]
        print(f"  Top 5 features: {', '.join(f'{k}({v:.3f})' for k, v in top5)}")

    # -- Save experiment artifacts --
    exp_cfg = config["experiments"]
    if not quiet:
        print(f"\n{'='*60}")
        print(f"SAVING EXPERIMENT ARTIFACTS")
        print(f"{'='*60}")

    exp_dir, ts = create_experiment_dir(exp_cfg["base_dir"])
    all_metrics["experiment"] = {"timestamp": ts, "directory": str(exp_dir)}

    if exp_cfg.get("save_config", True):
        cfg_path = save_experiment_config(exp_dir, config)
        if not quiet:
            print(f"  Config: {cfg_path}")

    if exp_cfg.get("save_metrics", True):
        met_path = save_experiment_metrics(exp_dir, all_metrics)
        if not quiet:
            print(f"  Metrics: {met_path}")

    if exp_cfg.get("save_model", True):
        # Get class labels from the final model
        labels = sorted(set(y_train))
        artifact = {
            "model": final_model,
            "feature_cols": feature_cols,
            "target_map": target_map,
            "class_labels": labels,
            "hyperparameters": config["model"]["params"],
            "random_state": config["model"]["random_state"],
            "experiment_timestamp": ts,
        }
        mdl_path = save_experiment_model(exp_dir, artifact)
        if not quiet:
            size_kb = mdl_path.stat().st_size / 1024
            print(f"  Model: {mdl_path} ({size_kb:.1f} KB)")

    # Update latest symlink
    latest = create_latest_symlink(Path(exp_cfg["base_dir"]), exp_dir)
    if latest and not quiet:
        print(f"  Latest symlink: {latest}")

    if not quiet:
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Experiment directory: {exp_dir}")

    return all_metrics


if __name__ == "__main__":
    run_pipeline()
