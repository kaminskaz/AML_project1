"""Smoke tests for Task 2 metric implementations.

Run:
  ./.venv/bin/python tests/smoke/test_metrics_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.metrics.classification import compute_metric, list_supported_metrics


def test_supported_metrics_nonempty() -> None:
    metrics = list_supported_metrics()
    assert isinstance(metrics, list)
    assert len(metrics) > 0


def test_basic_metrics_perfect_predictions() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.4, 0.6])

    for name in ["precision", "recall", "f1", "accuracy", "balanced_accuracy", "roc_auc", "pr_auc"]:
        res = compute_metric(name, y_true, y_proba, threshold=0.5)
        assert res.name in {
            "precision",
            "recall",
            "f1",
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc",
        }
        assert abs(res.value - 1.0) < 1e-12


def test_auc_degenerate_single_class_returns_nan() -> None:
    y_true = np.array([0, 0, 0, 0])
    y_proba = np.array([0.2, 0.3, 0.1, 0.4])

    roc = compute_metric("roc_auc", y_true, y_proba).value
    pr = compute_metric("pr_auc", y_true, y_proba).value
    bal = compute_metric("balanced_accuracy", y_true, y_proba).value

    assert np.isnan(roc)
    assert np.isnan(pr)
    assert np.isnan(bal)


if __name__ == "__main__":
    test_supported_metrics_nonempty()
    test_basic_metrics_perfect_predictions()
    test_auc_degenerate_single_class_returns_nan()
    print("metrics smoke tests: OK")
