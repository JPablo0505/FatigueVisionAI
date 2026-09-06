"""Comprehensive model evaluation and error analysis (S2-07).

Loads the trained model from the latest experiment, evaluates on the held-out
test set, performs feature importance analysis, and generates a detailed
evaluation report.

Usage::

    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model experiments/20260906T025051Z/model.joblib
    python scripts/evaluate_model.py --output-dir reports/

Outputs:
    reports/evaluation_report.md   — human-readable evaluation report
    reports/error_analysis.json    — structured error analysis
    reports/evaluation_metrics.json — machine-readable metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL = _REPO_ROOT / "experiments" / "latest" / "model.joblib"
_DEFAULT_DATA = _REPO_ROOT / "datasets" / "processed" / "yawdd_test.parquet"
_DEFAULT_OUTPUT = _REPO_ROOT / "reports"


def find_latest_model() -> Path:
    """Find the model.joblib in the most recent experiment directory."""
    experiments_dir = _REPO_ROOT / "experiments"
    if not experiments_dir.exists():
        raise FileNotFoundError(f"No experiments directory found at {experiments_dir}")
    # Find all timestamped directories
    dirs = sorted(
        [d for d in experiments_dir.iterdir() if d.is_dir() and d.name != "latest"],
        reverse=True,
    )
    for d in dirs:
        model_path = d / "model.joblib"
        if model_path.exists():
            return model_path
    raise FileNotFoundError("No model.joblib found in any experiment directory")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FatigueVisionAI Model Evaluation (S2-07)")
    parser.add_argument("--model", type=str, default=None, help="Path to model.joblib")
    parser.add_argument("--data", type=str, default=None, help="Path to test parquet")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for reports")
    parser.add_argument("--quiet", action="store_true", help="Suppress printed output")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------

def load_model(model_path: Path) -> dict[str, Any]:
    """Load the model artifact."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def prepare_data(
    df: pd.DataFrame, feature_cols: list[str], target_map: dict[str, str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Prepare features, labels, and predictions from a DataFrame."""
    X = df[feature_cols].values
    y_raw = df["fatigue_state"].values
    y = np.array([target_map.get(c, "medium") for c in y_raw])
    return X, y, y_raw, df


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> dict[str, Any]:
    """Compute comprehensive metrics from true/predicted labels."""
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    report_str = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    report_dict = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)

    per_class = {}
    for label in labels:
        if label in report_dict:
            per_class[label] = {
                "precision": round(report_dict[label]["precision"], 4),
                "recall": round(report_dict[label]["recall"], 4),
                "f1-score": round(report_dict[label]["f1-score"], 4),
                "support": int(report_dict[label]["support"]),
            }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_dict = {
        "labels": labels,
        "matrix": cm.tolist(),
        "description": "Rows = true labels, Columns = predicted labels",
    }

    return {
        "overall": {
            "accuracy": round(accuracy, 4),
            "precision_macro": round(precision_macro, 4),
            "recall_macro": round(recall_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "total_samples": int(len(y_true)),
        },
        "per_class": per_class,
        "confusion_matrix": cm_dict,
        "classification_report": report_str,
    }


def analyze_feature_importance(
    model: Any, feature_cols: list[str]
) -> list[dict[str, Any]]:
    """Extract and rank feature importances."""
    importances = model.feature_importances_
    ranked = sorted(
        [{"feature": name, "importance": round(float(imp), 4)}
         for name, imp in zip(feature_cols, importances)],
        key=lambda x: -x["importance"],
    )
    for i, feat in enumerate(ranked):
        feat["rank"] = i + 1
    return ranked


def analyze_errors(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_raw: np.ndarray,
    feature_cols: list[str],
) -> list[dict[str, Any]]:
    """Identify and analyze misclassified samples."""
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            row = df.iloc[i]
            error_info = {
                "index": int(i),
                "subject_id": str(row.get("subject_id", "unknown")),
                "video_file": str(row.get("video_file", "unknown")),
                "window_index": int(row.get("window_index", -1)),
                "true_label_raw": str(y_raw[i]),
                "true_label_mapped": str(y_true[i]),
                "predicted_label": str(y_pred[i]),
                "feature_values": {
                    feat: round(float(row[feat]), 4) for feat in feature_cols
                },
            }
            errors.append(error_info)
    return errors


def generate_confusion_matrix_text(cm_dict: dict[str, Any]) -> str:
    """Generate a readable confusion matrix as text."""
    labels = cm_dict["labels"]
    matrix = cm_dict["matrix"]
    max_label_len = max(len(l) for l in labels)
    header = " " * (max_label_len + 2) + "  ".join(f"{l:>{max_label_len}}" for l in labels)
    lines = [header]
    for i, label in enumerate(labels):
        row_str = f"{label:>{max_label_len}}  " + "  ".join(
            f"{matrix[i][j]:>{max_label_len}}" for j in range(len(labels))
        )
        lines.append(row_str)
    return "\n".join(lines)


def generate_markdown_report(
    metrics: dict[str, Any],
    feature_ranking: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    model_info: dict[str, Any],
    data_info: dict[str, Any],
) -> str:
    """Generate the full evaluation report as Markdown."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    overall = metrics["overall"]
    per_class = metrics["per_class"]
    cm = metrics["confusion_matrix"]

    lines = [
        "# FatigueVisionAI — Model Evaluation Report (S2-07)",
        "",
        f"**Generated**: {now}",
        f"**Model**: {model_info.get('classifier', 'RandomForestClassifier')}",
        f"**Test set**: {data_info['total_samples']} windows from {data_info['total_subjects']} subjects",
        f"**Hyperparameters**: {json.dumps(model_info.get('hyperparameters', {}), indent=2)}",
        "",
        "---",
        "",
        "## 1. Overall Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | **{overall['accuracy']:.4f}** |",
        f"| Precision (macro) | {overall['precision_macro']:.4f} |",
        f"| Recall (macro) | {overall['recall_macro']:.4f} |",
        f"| F1 Score (macro) | **{overall['f1_macro']:.4f}** |",
        f"| F1 Score (weighted) | **{overall['f1_weighted']:.4f}** |",
        f"| Total samples | {overall['total_samples']} |",
        "",
        "---",
        "",
        "## 2. Per-Class Metrics",
        "",
        "| Class | Precision | Recall | F1-Score | Support |",
        "|---|---|---|---|---|",
    ]

    for label, vals in per_class.items():
        lines.append(
            f"| {label} | {vals['precision']:.4f} | {vals['recall']:.4f} | "
            f"{vals['f1-score']:.4f} | {vals['support']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Confusion Matrix",
        "",
        "Rows = True labels, Columns = Predicted labels",
        "",
        "```",
        generate_confusion_matrix_text(cm),
        "```",
        "",
        "---",
        "",
        "## 4. Feature Importance Ranking",
        "",
        "| Rank | Feature | Importance |",
        "|---|---|---|",
    ])

    for feat in feature_ranking[:10]:
        lines.append(f"| {feat['rank']} | {feat['feature']} | {feat['importance']:.4f} |")

    lines.extend([
        "",
        "**Top 5 most important features**:",
    ])
    for feat in feature_ranking[:5]:
        lines.append(f"1. **{feat['feature']}** — {feat['importance']:.4f}")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Error Analysis",
        "",
        f"**Total misclassified**: {len(errors)} / {overall['total_samples']} "
        f"({len(errors) / overall['total_samples'] * 100:.1f}%)",
        "",
    ])

    if errors:
        # Group errors by confusion pair
        confusion_pairs: dict[str, list] = defaultdict(list)
        for err in errors:
            pair = f"{err['true_label_mapped']} → {err['predicted_label']}"
            confusion_pairs[pair].append(err)

        lines.append("### Confusion Pairs")
        lines.append("")
        for pair, samples in sorted(confusion_pairs.items(), key=lambda x: -len(x[1])):
            lines.append(f"**{pair}** ({len(samples)} samples):")
            lines.append("")
            for s in samples[:3]:  # Show up to 3 examples per pair
                lines.append(f"- Subject `{s['subject_id']}`, video `{s['video_file']}`, "
                           f"window {s['window_index']}")
                # Show top discriminative features
                fv = s["feature_values"]
                key_feats = ["perclos", "ear_mean", "mar_max", "yawn_mean", "blink_rate"]
                feat_str = ", ".join(f"{k}={fv.get(k, 'N/A')}" for k in key_feats if k in fv)
                lines.append(f"  - Features: {feat_str}")
            if len(samples) > 3:
                lines.append(f"  - ... and {len(samples) - 3} more")
            lines.append("")
    else:
        lines.append("**No misclassifications! Perfect score on test set.**")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 6. Limitations and Observations",
        "",
        "### Class Imbalance",
        f"- The test set has severe class imbalance: {data_info['class_distribution']}",
        "- The model was trained with `class_weight='balanced'` to mitigate this, but minority classes (yawning, unknown) may still be under-represented.",
        "",
        "### Data Constraints",
        "- Only 25 test windows from 8 subjects — results may not generalize to unseen populations.",
        "- YAWDD dataset is a controlled lab setting; real-world driving conditions may differ.",
        "",
        "### Feature Limitations",
        "- `ear_std` and `mar_std` are set to 0.0 in the adapter (FatigueFeatures provides window-level stats only).",
        "- The model does not use temporal dependencies (e.g., sequence of states over time).",
        "",
        "---",
        "",
        "*Report generated by FatigueVisionAI S2-07 evaluation pipeline.*",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation(argv: list[str] | None = None) -> dict[str, Any]:
    """Execute the full evaluation pipeline."""
    args = parse_args(argv)
    quiet = args.quiet

    # Resolve paths
    model_path = Path(args.model) if args.model else find_latest_model()
    data_path = Path(args.data) if args.data else _DEFAULT_DATA
    output_dir = Path(args.output_dir) if args.output_dir else _DEFAULT_OUTPUT

    if not quiet:
        print("=" * 60)
        print("FatigueVisionAI Model Evaluation (S2-07)")
        print("=" * 60)

    # Load model
    if not quiet:
        print(f"\nLoading model from {model_path} ...")
    artifact = load_model(model_path)
    model = artifact["model"]
    feature_cols: list[str] = artifact["feature_cols"]
    target_map: dict[str, str] = artifact["target_map"]
    hyperparams: dict = artifact.get("hyperparameters", {})

    if not quiet:
        print(f"  Model type: {type(model).__name__}")
        print(f"  Features: {len(feature_cols)}")

    # Load test data
    if not quiet:
        print(f"\nLoading test data from {data_path} ...")
    test_df = pd.read_parquet(data_path)
    if not quiet:
        print(f"  Windows: {len(test_df)}, Subjects: {test_df['subject_id'].nunique()}")

    # Prepare data
    X, y_mapped, y_raw, _ = prepare_data(test_df, feature_cols, target_map)
    if not quiet:
        print(f"  Label distribution: {dict(zip(*np.unique(y_mapped, return_counts=True)))}")

    # Predict
    if not quiet:
        print(f"\nRunning inference on {len(X)} samples ...")
    y_pred = model.predict(X)

    # Evaluate
    labels = sorted(set(y_mapped) | set(y_pred))
    metrics = evaluate_predictions(y_mapped, y_pred, labels)

    if not quiet:
        print(f"\n{'='*60}")
        print("OVERALL METRICS")
        print(f"{'='*60}")
        o = metrics["overall"]
        print(f"  Accuracy:      {o['accuracy']:.4f}")
        print(f"  Precision:     {o['precision_macro']:.4f}")
        print(f"  Recall:        {o['recall_macro']:.4f}")
        print(f"  F1 (macro):    {o['f1_macro']:.4f}")
        print(f"  F1 (weighted): {o['f1_weighted']:.4f}")
        print(f"\n{metrics['classification_report']}")

    # Feature importance
    feature_ranking = analyze_feature_importance(model, feature_cols)
    if not quiet:
        print(f"\n{'='*60}")
        print("FEATURE IMPORTANCE (Top 10)")
        print(f"{'='*60}")
        for feat in feature_ranking[:10]:
            bar = "#" * int(feat["importance"] * 50)
            print(f"  {feat['rank']:>2}. {feat['feature']:<30s} {feat['importance']:.4f} {bar}")

    # Error analysis
    errors = analyze_errors(test_df, y_mapped, y_pred, y_raw, feature_cols)
    if not quiet:
        print(f"\n{'='*60}")
        print(f"ERROR ANALYSIS: {len(errors)} / {len(y_mapped)} misclassified")
        print(f"{'='*60}")
        if errors:
            for err in errors:
                print(f"  [{err['subject_id']}] {err['true_label_mapped']} → "
                      f"{err['predicted_label']} (video: {err['video_file']})")
        else:
            print("  No misclassifications!")

    # Model info
    model_info = {
        "classifier": type(model).__name__,
        "hyperparameters": hyperparams,
        "feature_count": len(feature_cols),
        "class_labels": artifact.get("class_labels", []),
    }

    # Data info
    data_info = {
        "total_samples": len(test_df),
        "total_subjects": int(test_df["subject_id"].nunique()),
        "class_distribution": dict(test_df["fatigue_state"].value_counts()),
    }

    # Generate reports
    output_dir.mkdir(parents=True, exist_ok=True)

    # Markdown report
    md_report = generate_markdown_report(metrics, feature_ranking, errors, model_info, data_info)
    md_path = output_dir / "evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    if not quiet:
        print(f"\nReport saved to {md_path}")

    # Error analysis JSON
    error_data = {
        "model_path": str(model_path),
        "test_data_path": str(data_path),
        "total_samples": len(y_mapped),
        "total_errors": len(errors),
        "error_rate": round(len(errors) / len(y_mapped), 4),
        "errors": errors,
        "confusion_pairs_summary": {},
    }
    # Summarize confusion pairs
    for err in errors:
        pair = f"{err['true_label_mapped']}→{err['predicted_label']}"
        error_data["confusion_pairs_summary"][pair] = (
            error_data["confusion_pairs_summary"].get(pair, 0) + 1
        )
    error_path = output_dir / "error_analysis.json"
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(error_data, f, indent=2, default=str)
    if not quiet:
        print(f"Error analysis saved to {error_path}")

    # Metrics JSON
    full_metrics = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "test_data_path": str(data_path),
        "model_info": model_info,
        "data_info": data_info,
        "metrics": metrics,
        "feature_importance": feature_ranking,
    }
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2, default=str)
    if not quiet:
        print(f"Metrics saved to {metrics_path}")

    if not quiet:
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}")

    return full_metrics


if __name__ == "__main__":
    run_evaluation()
