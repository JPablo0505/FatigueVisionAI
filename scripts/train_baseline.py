"""Train a baseline RandomForest fatigue classifier.

Usage::

    python scripts/train_baseline.py

Reads yawdd_train.parquet and yawdd_val.parquet, trains a
RandomForestClassifier, evaluates on validation, and exports:
- packages/ai-engine/models/baseline_fatigue.joblib (model artifact)
- datasets/processed/baseline_metrics.json (validation metrics)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Feature columns (20 numeric features from S2-03 windowing)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "ear_mean", "ear_std", "ear_min", "ear_max",
    "mar_mean", "mar_std", "mar_min", "mar_max",
    "perclos",
    "blink_rate",
    "eye_closure_mean", "eye_closure_max",
    "yawn_mean", "yawn_max", "yawn_fraction",
    "head_pose_yaw_mean", "head_pose_pitch_mean", "head_pose_roll_mean",
    "iris_x_mean", "iris_y_mean",
]

# ---------------------------------------------------------------------------
# Target mapping: fatigue_state (YAWDD) -> FatigueLevel
# ---------------------------------------------------------------------------
# normal/talking -> low (alert), unknown -> medium, yawning -> high (fatigued)
TARGET_MAP = {
    "normal": "low",
    "talking": "low",
    "unknown": "medium",
    "yawning": "high",
}

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2


def main() -> None:
    train_path = Path("datasets/processed/yawdd_train.parquet")
    val_path = Path("datasets/processed/yawdd_val.parquet")
    model_dir = Path("packages/ai-engine/models")
    model_path = model_dir / "baseline_fatigue.joblib"
    metrics_path = Path("datasets/processed/baseline_metrics.json")

    # -- Load data --
    print(f"Loading training data from {train_path} ...")
    train_df = pd.read_parquet(train_path)
    print(f"  Train windows: {len(train_df)}")

    print(f"Loading validation data from {val_path} ...")
    val_df = pd.read_parquet(val_path)
    print(f"  Val windows: {len(val_df)}")

    # -- Prepare features --
    X_train = train_df[FEATURE_COLS].values
    X_val = val_df[FEATURE_COLS].values

    y_train_raw = train_df["fatigue_state"].values
    y_val_raw = val_df["fatigue_state"].values

    # Map to FatigueLevel
    y_train = np.array([TARGET_MAP.get(c, "medium") for c in y_train_raw])
    y_val = np.array([TARGET_MAP.get(c, "medium") for c in y_val_raw])

    print(f"\nTarget mapping: {TARGET_MAP}")
    print(f"Train labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Val labels:   {dict(zip(*np.unique(y_val, return_counts=True)))}")

    # -- Train model --
    print(f"\nTraining RandomForestClassifier (n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}) ...")
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.2f}s")

    # -- Evaluate on validation --
    print(f"\nEvaluating on validation set ({len(X_val)} windows) ...")
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    precision_macro = precision_score(y_val, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_val, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"VALIDATION METRICS")
    print(f"{'='*60}")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision (macro):  {precision_macro:.4f}")
    print(f"Recall (macro):     {recall_macro:.4f}")
    print(f"F1 (macro):         {f1_macro:.4f}")
    print(f"F1 (weighted):      {f1_weighted:.4f}")

    # Classification report
    labels = sorted(set(y_val) | set(y_pred))
    report = classification_report(y_val, y_pred, labels=labels, zero_division=0)
    print(f"\nClassification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred, labels=labels)
    print(f"Confusion Matrix (rows=true, cols=pred):")
    print(f"  Labels: {labels}")
    for i, label in enumerate(labels):
        print(f"  {label:<10s}: {cm[i].tolist()}")

    # Feature importances
    importances = clf.feature_importances_
    top_k = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])[:5]
    print(f"\nTop 5 feature importances:")
    for name, imp in top_k:
        print(f"  {name:<30s} {imp:.4f}")

    # -- Save model artifact --
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": clf,
        "feature_cols": FEATURE_COLS,
        "target_map": TARGET_MAP,
        "class_labels": labels,
        "hyperparameters": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_split": MIN_SAMPLES_SPLIT,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
        },
    }
    joblib.dump(artifact, model_path)
    print(f"\nModel artifact saved to {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

    # -- Save metrics --
    metrics = {
        "model": "RandomForestClassifier",
        "hyperparameters": artifact["hyperparameters"],
        "train_windows": len(train_df),
        "val_windows": len(val_df),
        "train_time_seconds": round(train_time, 2),
        "feature_count": len(FEATURE_COLS),
        "feature_cols": FEATURE_COLS,
        "target_mapping": TARGET_MAP,
        "validation_metrics": {
            "accuracy": round(accuracy, 4),
            "precision_macro": round(precision_macro, 4),
            "recall_macro": round(recall_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
        },
        "classification_report": report,
        "confusion_matrix": {str(label): cm[i].tolist() for i, label in enumerate(labels)},
        "confusion_matrix_labels": labels,
        "feature_importances": {name: round(float(imp), 4) for name, imp in zip(FEATURE_COLS, importances)},
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
