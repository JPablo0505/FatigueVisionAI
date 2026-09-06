"""Unit tests for the S2-07 model evaluation and error analysis.

Tests evaluation metrics computation, feature importance extraction,
error analysis logic, report generation, and end-to-end evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from evaluate_model import (
    analyze_errors,
    analyze_feature_importance,
    evaluate_predictions,
    generate_confusion_matrix_text,
    generate_markdown_report,
    load_model,
)

_MODEL_PATH = _REPO_ROOT / "experiments" / "20260906T025051Z" / "model.joblib"
_TEST_PARQUET = _REPO_ROOT / "datasets" / "processed" / "yawdd_test.parquet"


@pytest.fixture(scope="module")
def model_artifact() -> dict:
    if not _MODEL_PATH.exists():
        pytest.skip("Model artifact not found")
    return load_model(_MODEL_PATH)


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    if not _TEST_PARQUET.exists():
        pytest.skip("Test data not found")
    return pd.read_parquet(_TEST_PARQUET)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

class TestEvaluatePredictions:
    def test_perfect_prediction(self) -> None:
        y = np.array(["low", "low", "high", "high"])
        result = evaluate_predictions(y, y, ["high", "low"])
        assert result["overall"]["accuracy"] == 1.0
        assert result["overall"]["f1_macro"] == 1.0
        assert result["overall"]["total_samples"] == 4

    def test_imperfect_prediction(self) -> None:
        y_true = np.array(["low", "low", "high", "high"])
        y_pred = np.array(["low", "high", "high", "high"])
        result = evaluate_predictions(y_true, y_pred, ["high", "low"])
        assert result["overall"]["accuracy"] == 0.75
        assert result["overall"]["f1_macro"] < 1.0

    def test_per_class_structure(self) -> None:
        y = np.array(["low", "medium", "high"])
        result = evaluate_predictions(y, y, ["high", "low", "medium"])
        assert "high" in result["per_class"]
        assert "low" in result["per_class"]
        assert "medium" in result["per_class"]
        for cls in ["high", "low", "medium"]:
            assert "precision" in result["per_class"][cls]
            assert "recall" in result["per_class"][cls]
            assert "f1-score" in result["per_class"][cls]
            assert "support" in result["per_class"][cls]

    def test_confusion_matrix_structure(self) -> None:
        y = np.array(["low", "high", "low", "high"])
        result = evaluate_predictions(y, y, ["high", "low"])
        cm = result["confusion_matrix"]
        assert "labels" in cm
        assert "matrix" in cm
        assert len(cm["matrix"]) == 2
        assert len(cm["matrix"][0]) == 2
        # Diagonal should be all correct
        assert cm["matrix"][0][0] == 2  # high correctly predicted
        assert cm["matrix"][1][1] == 2  # low correctly predicted

    def test_zero_samples_returns_zero_metrics(self) -> None:
        y_true = np.array(["low"])
        y_pred = np.array(["high"])
        result = evaluate_predictions(y_true, y_pred, ["high", "low"])
        assert result["overall"]["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

class TestFeatureImportance:
    def test_returns_ranked_list(self, model_artifact: dict) -> None:
        model = model_artifact["model"]
        feature_cols = model_artifact["feature_cols"]
        ranking = analyze_feature_importance(model, feature_cols)
        assert len(ranking) == 20
        assert ranking[0]["rank"] == 1
        assert ranking[-1]["rank"] == 20

    def test_sorted_by_importance_desc(self, model_artifact: dict) -> None:
        model = model_artifact["model"]
        feature_cols = model_artifact["feature_cols"]
        ranking = analyze_feature_importance(model, feature_cols)
        importances = [f["importance"] for f in ranking]
        assert importances == sorted(importances, reverse=True)

    def test_all_importances_non_negative(self, model_artifact: dict) -> None:
        model = model_artifact["model"]
        feature_cols = model_artifact["feature_cols"]
        ranking = analyze_feature_importance(model, feature_cols)
        for feat in ranking:
            assert feat["importance"] >= 0.0

    def test_importances_sum_to_one(self, model_artifact: dict) -> None:
        model = model_artifact["model"]
        feature_cols = model_artifact["feature_cols"]
        ranking = analyze_feature_importance(model, feature_cols)
        total = sum(f["importance"] for f in ranking)
        assert abs(total - 1.0) < 0.01  # RandomForest importances sum to ~1.0


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

class TestErrorAnalysis:
    def test_no_errors_when_correct(self) -> None:
        df = pd.DataFrame({
            "subject_id": ["1", "2"],
            "video_file": ["a.mp4", "b.mp4"],
            "window_index": [0, 0],
            "ear_mean": [0.3, 0.3],
        })
        y_true = np.array(["low", "high"])
        y_pred = np.array(["low", "high"])
        y_raw = np.array(["normal", "yawning"])
        errors = analyze_errors(df, y_true, y_pred, y_raw, ["ear_mean"])
        assert len(errors) == 0

    def test_captures_errors(self) -> None:
        df = pd.DataFrame({
            "subject_id": ["1", "2", "3"],
            "video_file": ["a.mp4", "b.mp4", "c.mp4"],
            "window_index": [0, 1, 2],
            "ear_mean": [0.3, 0.2, 0.4],
        })
        y_true = np.array(["low", "high", "medium"])
        y_pred = np.array(["low", "medium", "medium"])
        y_raw = np.array(["normal", "yawning", "unknown"])
        errors = analyze_errors(df, y_true, y_pred, y_raw, ["ear_mean"])
        assert len(errors) == 1
        assert errors[0]["subject_id"] == "2"
        assert errors[0]["true_label_mapped"] == "high"
        assert errors[0]["predicted_label"] == "medium"

    def test_error_includes_feature_values(self) -> None:
        df = pd.DataFrame({
            "subject_id": ["1"],
            "video_file": ["a.mp4"],
            "window_index": [0],
            "ear_mean": [0.35],
            "perclos": [0.1],
        })
        y_true = np.array(["low"])
        y_pred = np.array(["high"])
        y_raw = np.array(["normal"])
        errors = analyze_errors(df, y_true, y_pred, y_raw, ["ear_mean", "perclos"])
        assert len(errors) == 1
        assert errors[0]["feature_values"]["ear_mean"] == 0.35
        assert errors[0]["feature_values"]["perclos"] == 0.1


# ---------------------------------------------------------------------------
# Confusion matrix text
# ---------------------------------------------------------------------------

class TestConfusionMatrixText:
    def test_generates_readable_output(self) -> None:
        cm_dict = {
            "labels": ["high", "low"],
            "matrix": [[5, 0], [1, 9]],
        }
        text = generate_confusion_matrix_text(cm_dict)
        assert "high" in text
        assert "low" in text
        assert "5" in text
        assert "9" in text


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

class TestMarkdownReport:
    def test_generates_valid_markdown(self, model_artifact: dict) -> None:
        metrics = {
            "overall": {
                "accuracy": 1.0, "precision_macro": 1.0,
                "recall_macro": 1.0, "f1_macro": 1.0,
                "f1_weighted": 1.0, "total_samples": 25,
            },
            "per_class": {
                "low": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 22},
                "medium": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 2},
                "high": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 1},
            },
            "confusion_matrix": {
                "labels": ["high", "low", "medium"],
                "matrix": [[1, 0, 0], [0, 22, 0], [0, 0, 2]],
            },
            "classification_report": "test report",
        }
        feature_ranking = [
            {"feature": "ear_mean", "importance": 0.15, "rank": 1},
            {"feature": "perclos", "importance": 0.10, "rank": 2},
        ]
        errors = []
        model_info = {"classifier": "RandomForest", "hyperparameters": {}, "feature_count": 20, "class_labels": []}
        data_info = {"total_samples": 25, "total_subjects": 8, "class_distribution": {}}

        md = generate_markdown_report(metrics, feature_ranking, errors, model_info, data_info)
        assert "# FatigueVisionAI" in md
        assert "## 1. Overall Metrics" in md
        assert "## 4. Feature Importance Ranking" in md
        assert "## 5. Error Analysis" in md
        assert "No misclassifications" in md

    def test_report_includes_errors_when_present(self) -> None:
        metrics = {
            "overall": {"accuracy": 0.8, "precision_macro": 0.8, "recall_macro": 0.8,
                        "f1_macro": 0.8, "f1_weighted": 0.8, "total_samples": 10},
            "per_class": {},
            "confusion_matrix": {"labels": ["low", "high"], "matrix": [[8, 1], [1, 0]]},
            "classification_report": "",
        }
        errors = [{
            "subject_id": "1", "video_file": "a.mp4", "window_index": 0,
            "true_label_mapped": "low", "predicted_label": "high",
            "true_label_raw": "normal", "feature_values": {"perclos": 0.1},
        }]
        md = generate_markdown_report(metrics, [], errors, {}, {"total_samples": 10, "total_subjects": 2, "class_distribution": {}})
        assert "1 / 10" in md
        assert "low → high" in md


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class TestModelLoading:
    def test_load_returns_artifact(self, model_artifact: dict) -> None:
        assert "model" in model_artifact
        assert "feature_cols" in model_artifact
        assert "target_map" in model_artifact
        assert len(model_artifact["feature_cols"]) == 20

    def test_invalid_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_model(Path("/nonexistent/model.joblib"))
