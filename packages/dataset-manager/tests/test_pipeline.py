"""Unit tests for the S2-06 reproducible training pipeline.

Tests pipeline reproducibility (same seed → same results), config loading,
experiment directory creation, GroupKFold CV correctness, and CLI overrides.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the pipeline module
# ---------------------------------------------------------------------------
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from train_pipeline import (
    apply_cli_overrides,
    compute_metrics,
    create_experiment_dir,
    load_config,
    parse_args,
    run_group_kfold_cv,
    run_holdout_eval,
    run_pipeline,
    save_experiment_config,
    save_experiment_metrics,
    save_experiment_model,
    train_model,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PROCESSED = _REPO_ROOT / "datasets" / "processed"
_TRAIN_PARQUET = _PROCESSED / "yawdd_train.parquet"
_VAL_PARQUET = _PROCESSED / "yawdd_val.parquet"
_TEST_PARQUET = _PROCESSED / "yawdd_test.parquet"
_CONFIG_PATH = _REPO_ROOT / "config" / "train_config.yaml"


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    if not _TRAIN_PARQUET.exists():
        pytest.skip("Training data not found")
    return pd.read_parquet(_TRAIN_PARQUET)


@pytest.fixture(scope="module")
def val_df() -> pd.DataFrame:
    if not _VAL_PARQUET.exists():
        pytest.skip("Validation data not found")
    return pd.read_parquet(_VAL_PARQUET)


@pytest.fixture(scope="module")
def sample_config() -> dict:
    return load_config(_CONFIG_PATH)


@pytest.fixture()
def tmp_experiments(tmp_path: Path) -> Path:
    return tmp_path / "experiments"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_load_default_config(self) -> None:
        config = load_config(None)
        assert "data" in config
        assert "model" in config
        assert "evaluation" in config
        assert "features" in config

    def test_load_yaml_config(self) -> None:
        if not _CONFIG_PATH.exists():
            pytest.skip("Config file not found")
        config = load_config(_CONFIG_PATH)
        assert config["evaluation"]["strategy"] == "group_kfold"
        assert config["evaluation"]["group_kfold"]["n_splits"] == 5

    def test_missing_config_uses_defaults(self) -> None:
        config = load_config(Path("/nonexistent/config.yaml"))
        assert config["model"]["random_state"] == 42
        assert config["evaluation"]["strategy"] == "group_kfold"

    def test_config_deep_merge(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  params:\n    n_estimators: 200\n")
            f.flush()
            config = load_config(f.name)
        assert config["model"]["params"]["n_estimators"] == 200
        # Other defaults preserved
        assert config["model"]["params"]["max_depth"] == 10


# ---------------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------------

class TestCLIOverrides:
    def test_parse_args_defaults(self) -> None:
        args = parse_args([])
        assert args.config is None
        assert args.strategy is None
        assert args.n_estimators is None

    def test_apply_cli_overrides(self) -> None:
        config = load_config(None)
        args = parse_args(["--n-estimators", "200", "--max-depth", "5", "--random-state", "123"])
        config = apply_cli_overrides(config, args)
        assert config["model"]["params"]["n_estimators"] == 200
        assert config["model"]["params"]["max_depth"] == 5
        assert config["model"]["random_state"] == 123

    def test_apply_strategy_override(self) -> None:
        config = load_config(None)
        args = parse_args(["--strategy", "holdout"])
        config = apply_cli_overrides(config, args)
        assert config["evaluation"]["strategy"] == "holdout"


# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------

class TestExperimentTracking:
    def test_create_experiment_dir(self, tmp_experiments: Path) -> None:
        exp_dir, ts = create_experiment_dir(tmp_experiments)
        assert exp_dir.exists()
        assert exp_dir.is_dir()
        assert ts in str(exp_dir)

    def test_save_config_json(self, tmp_experiments: Path) -> None:
        exp_dir, _ = create_experiment_dir(tmp_experiments)
        config = {"test": True, "value": 42}
        path = save_experiment_config(exp_dir, config)
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["test"] is True
        assert loaded["value"] == 42

    def test_save_metrics_json(self, tmp_experiments: Path) -> None:
        exp_dir, _ = create_experiment_dir(tmp_experiments)
        metrics = {"accuracy": 0.95, "f1_macro": 0.88}
        path = save_experiment_metrics(exp_dir, metrics)
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["accuracy"] == 0.95

    def test_save_model_joblib(self, tmp_experiments: Path) -> None:
        import joblib
        exp_dir, _ = create_experiment_dir(tmp_experiments)
        artifact = {"model": "dummy", "feature_cols": ["a", "b"]}
        path = save_experiment_model(exp_dir, artifact)
        assert path.exists()
        assert path.stat().st_size > 0
        loaded = joblib.load(path)
        assert loaded["model"] == "dummy"


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

class TestMetricsComputation:
    def test_perfect_prediction(self) -> None:
        y = np.array(["low", "low", "high", "high"])
        metrics = compute_metrics(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0
        assert metrics["f1_weighted"] == 1.0

    def test_known_imperfect_prediction(self) -> None:
        y_true = np.array(["low", "low", "high", "high"])
        y_pred = np.array(["low", "high", "high", "high"])
        metrics = compute_metrics(y_true, y_pred)
        assert 0.0 < metrics["accuracy"] < 1.0
        assert 0.0 < metrics["f1_macro"] <= 1.0


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

class TestModelTraining:
    def test_train_returns_classifier(self, train_df: pd.DataFrame, sample_config: dict) -> None:
        from sklearn.ensemble import RandomForestClassifier
        feature_cols = sample_config["features"]["columns"]
        target_map = sample_config["features"]["target_map"]
        X = train_df[feature_cols].values
        y = np.array([target_map.get(c, "medium") for c in train_df["fatigue_state"].values])
        clf = train_model(X, y, sample_config)
        assert isinstance(clf, RandomForestClassifier)
        assert hasattr(clf, "predict")

    def test_deterministic_training(self, train_df: pd.DataFrame, sample_config: dict) -> None:
        """Same seed + same data = same model predictions."""
        feature_cols = sample_config["features"]["columns"]
        target_map = sample_config["features"]["target_map"]
        X = train_df[feature_cols].values
        y = np.array([target_map.get(c, "medium") for c in train_df["fatigue_state"].values])

        clf1 = train_model(X, y, sample_config)
        clf2 = train_model(X, y, sample_config)

        preds1 = clf1.predict(X)
        preds2 = clf2.predict(X)
        np.testing.assert_array_equal(preds1, preds2)


# ---------------------------------------------------------------------------
# GroupKFold CV correctness
# ---------------------------------------------------------------------------

class TestGroupKFoldCV:
    def test_cv_returns_expected_structure(self, train_df: pd.DataFrame, sample_config: dict) -> None:
        result = run_group_kfold_cv(
            train_df,
            sample_config["features"]["columns"],
            sample_config["features"]["target_map"],
            "subject_id",
            sample_config,
            quiet=True,
        )
        assert result["strategy"] == "group_kfold"
        assert result["n_splits"] == 5
        assert len(result["fold_details"]) == 5
        assert "aggregated" in result
        assert "f1_macro_mean" in result["aggregated"]
        assert "f1_macro_std" in result["aggregated"]

    def test_cv_no_subject_leakage(self, train_df: pd.DataFrame, sample_config: dict) -> None:
        """No subject appears in both train and val within the same fold."""
        result = run_group_kfold_cv(
            train_df,
            sample_config["features"]["columns"],
            sample_config["features"]["target_map"],
            "subject_id",
            sample_config,
            quiet=True,
        )
        for fold in result["fold_details"]:
            val_ids = set(fold["val_subject_ids"])
            # All train subjects = total subjects - val subjects
            total = train_df["subject_id"].nunique()
            train_count = fold["train_subjects"]
            assert train_count + len(val_ids) == total

    def test_cv_metrics_in_valid_range(self, train_df: pd.DataFrame, sample_config: dict) -> None:
        result = run_group_kfold_cv(
            train_df,
            sample_config["features"]["columns"],
            sample_config["features"]["target_map"],
            "subject_id",
            sample_config,
            quiet=True,
        )
        agg = result["aggregated"]
        for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]:
            mean = agg[f"{key}_mean"]
            std = agg[f"{key}_std"]
            assert 0.0 <= mean <= 1.0, f"{key}_mean out of range: {mean}"
            assert std >= 0.0, f"{key}_std negative: {std}"

    def test_cv_deterministic_with_same_seed(
        self, train_df: pd.DataFrame, sample_config: dict
    ) -> None:
        """Same config → same CV metrics."""
        r1 = run_group_kfold_cv(
            train_df,
            sample_config["features"]["columns"],
            sample_config["features"]["target_map"],
            "subject_id",
            sample_config,
            quiet=True,
        )
        r2 = run_group_kfold_cv(
            train_df,
            sample_config["features"]["columns"],
            sample_config["features"]["target_map"],
            "subject_id",
            sample_config,
            quiet=True,
        )
        assert r1["aggregated"]["f1_macro_mean"] == r2["aggregated"]["f1_macro_mean"]
        assert r1["aggregated"]["f1_weighted_mean"] == r2["aggregated"]["f1_weighted_mean"]


# ---------------------------------------------------------------------------
# Holdout evaluation
# ---------------------------------------------------------------------------

class TestHoldoutEval:
    def test_holdout_returns_expected_structure(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, sample_config: dict
    ) -> None:
        feature_cols = sample_config["features"]["columns"]
        target_map = sample_config["features"]["target_map"]
        X_train = train_df[feature_cols].values
        y_train = np.array([target_map.get(c, "medium") for c in train_df["fatigue_state"].values])
        X_val = val_df[feature_cols].values
        y_val = np.array([target_map.get(c, "medium") for c in val_df["fatigue_state"].values])

        result = run_holdout_eval(X_train, y_train, X_val, y_val, sample_config)
        assert "metrics" in result
        assert result["samples"] == len(val_df)
        assert 0.0 <= result["metrics"]["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Full pipeline reproducibility
# ---------------------------------------------------------------------------

class TestPipelineReproducibility:
    def test_full_pipeline_run(self, tmp_experiments: Path) -> None:
        """Run the full pipeline with a temp experiments dir and verify outputs."""
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        metrics = run_pipeline([
            "--config", str(repo_root / "config" / "train_config.yaml"),
            "--experiments-dir", str(tmp_experiments),
            "--train-path", str(repo_root / "datasets" / "processed" / "yawdd_train.parquet"),
            "--val-path", str(repo_root / "datasets" / "processed" / "yawdd_val.parquet"),
            "--test-path", str(repo_root / "datasets" / "processed" / "yawdd_test.parquet"),
            "--no-holdout-test",
        ])
        assert "cross_validation" in metrics
        assert "final_model" in metrics
        assert "experiment" in metrics

        exp_dir = Path(metrics["experiment"]["directory"])
        assert (exp_dir / "config.json").exists()
        assert (exp_dir / "metrics.json").exists()
        assert (exp_dir / "model.joblib").exists()

    def test_pipeline_reproducible_results(self, tmp_experiments: Path) -> None:
        """Two runs with same config produce identical CV metrics."""
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        base_args = [
            "--config", str(repo_root / "config" / "train_config.yaml"),
            "--train-path", str(repo_root / "datasets" / "processed" / "yawdd_train.parquet"),
            "--val-path", str(repo_root / "datasets" / "processed" / "yawdd_val.parquet"),
            "--test-path", str(repo_root / "datasets" / "processed" / "yawdd_test.parquet"),
            "--no-holdout-test", "--quiet",
        ]
        r1 = run_pipeline(base_args + ["--experiments-dir", str(tmp_experiments / "run1")])
        r2 = run_pipeline(base_args + ["--experiments-dir", str(tmp_experiments / "run2")])
        assert r1["cross_validation"]["aggregated"]["f1_macro_mean"] == \
               r2["cross_validation"]["aggregated"]["f1_macro_mean"]
        assert r1["cross_validation"]["aggregated"]["f1_weighted_mean"] == \
               r2["cross_validation"]["aggregated"]["f1_weighted_mean"]
