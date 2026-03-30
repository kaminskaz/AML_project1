from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


MetricName = Literal[
    "precision",
    "recall",
    "f1",
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
]


@dataclass(frozen=True)
class MetricResult:
    """Result of computing a metric.

    `value` is a float; may be `np.nan` if undefined (e.g., AUC with one class).
    """

    name: str
    value: float


def _as_1d_float_array(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        arr = np.ravel(arr)
    return arr


def _as_1d_int_array(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=int)
    if arr.ndim != 1:
        arr = np.ravel(arr)
    return arr


def _check_binary_labels(y_true: np.ndarray) -> None:
    unique = np.unique(y_true)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(
            "Task 2 metrics expect y_true in {0,1}. "
            f"Found labels: {unique.tolist()}"
        )


def _check_proba(y_proba: np.ndarray) -> None:
    if np.any(np.isnan(y_proba)):
        raise ValueError("y_proba contains NaNs")
    if np.any((y_proba < 0.0) | (y_proba > 1.0)):
        raise ValueError("y_proba must be in [0,1]")


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    _check_binary_labels(y_true)
    _check_binary_labels(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def precision(y_true: ArrayLike, y_pred: ArrayLike, *, zero_division: float = 0.0) -> float:
    """Precision = TP / (TP + FP).

    If TP+FP == 0, returns `zero_division`.
    """

    yt = _as_1d_int_array(y_true)
    yp = _as_1d_int_array(y_pred)
    tp, _, fp, _ = _confusion_counts(yt, yp)
    denom = tp + fp
    return float(tp / denom) if denom > 0 else float(zero_division)


def recall(y_true: ArrayLike, y_pred: ArrayLike, *, zero_division: float = 0.0) -> float:
    """Recall (sensitivity) = TP / (TP + FN).

    If TP+FN == 0, returns `zero_division`.
    """

    yt = _as_1d_int_array(y_true)
    yp = _as_1d_int_array(y_pred)
    tp, _, _, fn = _confusion_counts(yt, yp)
    denom = tp + fn
    return float(tp / denom) if denom > 0 else float(zero_division)


def f1(y_true: ArrayLike, y_pred: ArrayLike, *, zero_division: float = 0.0) -> float:
    """F1 = 2 * (precision * recall) / (precision + recall).

    If precision+recall == 0, returns `zero_division`.
    """

    p = precision(y_true, y_pred, zero_division=zero_division)
    r = recall(y_true, y_pred, zero_division=zero_division)
    denom = p + r
    return float(2.0 * p * r / denom) if denom > 0 else float(zero_division)


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Accuracy = (TP + TN) / (TP + TN + FP + FN)."""

    yt = _as_1d_int_array(y_true)
    yp = _as_1d_int_array(y_pred)
    _confusion_counts(yt, yp)  # validates binary labels
    return float(np.mean(yt == yp))


def balanced_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Balanced accuracy at a fixed threshold.

    Balanced accuracy = (TPR + TNR) / 2.

    Returns `np.nan` if either class is missing in `y_true`.
    """

    yt = _as_1d_int_array(y_true)
    yp = _as_1d_int_array(y_pred)
    tp, tn, fp, fn = _confusion_counts(yt, yp)

    pos = tp + fn
    neg = tn + fp
    if pos == 0 or neg == 0:
        return float(np.nan)

    tpr = tp / pos
    tnr = tn / neg
    return float(0.5 * (tpr + tnr))


def roc_auc(y_true: ArrayLike, y_proba: ArrayLike) -> float:
    """ROC AUC using probability scores for class 1.

    Returns `np.nan` if undefined (e.g., only one class present).
    """

    yt = _as_1d_int_array(y_true)
    yp = _as_1d_float_array(y_proba)
    _check_binary_labels(yt)
    _check_proba(yp)

    if np.unique(yt).size < 2:
        return float(np.nan)

    try:
        return float(roc_auc_score(yt, yp))
    except ValueError:
        return float(np.nan)


def pr_auc(y_true: ArrayLike, y_proba: ArrayLike) -> float:
    """Area under the sensitivity–precision curve (interpreted as PR AUC).

    Uses recall on the x-axis and precision on the y-axis.

    Returns `np.nan` if undefined (e.g., only one class present).
    """

    yt = _as_1d_int_array(y_true)
    yp = _as_1d_float_array(y_proba)
    _check_binary_labels(yt)
    _check_proba(yp)

    if np.unique(yt).size < 2:
        return float(np.nan)

    try:
        prec, rec, _ = precision_recall_curve(yt, yp)
    except ValueError:
        return float(np.nan)

    # `rec` is non-decreasing; integrate precision over recall.
    return float(auc(rec, prec))


def threshold_predictions(y_proba: ArrayLike, *, threshold: float = 0.5) -> np.ndarray:
    """Convert probabilities into hard labels using a fixed threshold."""

    yp = _as_1d_float_array(y_proba)
    _check_proba(yp)
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be in (0,1)")
    return (yp >= threshold).astype(int)


_ALIASES: dict[str, MetricName] = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "f-measure": "f1",
    "f_measure": "f1",
    "fscore": "f1",
    "accuracy": "accuracy",
    "acc": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "balanced-accuracy": "balanced_accuracy",
    "roc_auc": "roc_auc",
    "roc-auc": "roc_auc",
    "auc": "roc_auc",
    "pr_auc": "pr_auc",
    "pr-auc": "pr_auc",
    "sensitivity_precision_auc": "pr_auc",
    "area under the sensitivity–precision curve": "pr_auc",
    "area under the sensitivity-precision curve": "pr_auc",
}


def list_supported_metrics() -> list[str]:
    """Return a list of supported metric names (including aliases)."""

    return sorted(_ALIASES.keys())


def get_metric_callable(metric: str) -> Callable[..., float]:
    """Return a callable that computes the named metric.

    The returned callable has signature:
    - threshold metrics: `(y_true, y_proba, threshold=0.5) -> float`
    - AUC metrics: `(y_true, y_proba, threshold=0.5) -> float` (threshold ignored)
    """

    key = metric.strip().lower()
    if key not in _ALIASES:
        raise KeyError(
            f"Unknown metric '{metric}'. Supported: {', '.join(list_supported_metrics())}"
        )

    canonical = _ALIASES[key]

    if canonical == "precision":
        return lambda y_true, y_proba, threshold=0.5: precision(
            y_true, threshold_predictions(y_proba, threshold=threshold)
        )
    if canonical == "recall":
        return lambda y_true, y_proba, threshold=0.5: recall(
            y_true, threshold_predictions(y_proba, threshold=threshold)
        )
    if canonical == "f1":
        return lambda y_true, y_proba, threshold=0.5: f1(
            y_true, threshold_predictions(y_proba, threshold=threshold)
        )
    if canonical == "accuracy":
        return lambda y_true, y_proba, threshold=0.5: accuracy(
            y_true, threshold_predictions(y_proba, threshold=threshold)
        )
    if canonical == "balanced_accuracy":
        return lambda y_true, y_proba, threshold=0.5: balanced_accuracy(
            y_true, threshold_predictions(y_proba, threshold=threshold)
        )
    if canonical == "roc_auc":
        return lambda y_true, y_proba, threshold=0.5: roc_auc(y_true, y_proba)
    if canonical == "pr_auc":
        return lambda y_true, y_proba, threshold=0.5: pr_auc(y_true, y_proba)

    raise RuntimeError("Unhandled metric mapping")


def compute_metric(metric: str, y_true: ArrayLike, y_proba: ArrayLike, *, threshold: float = 0.5) -> MetricResult:
    """Compute a validation metric for binary classification.

    Parameters
    - metric: name or alias of the metric
    - y_true: true labels in {0,1}
    - y_proba: probabilities for class 1, in [0,1]
    - threshold: threshold used for threshold-based metrics

    Returns
    - MetricResult(name=<canonical_metric_name>, value=<float>)
    """

    key = metric.strip().lower()
    if key not in _ALIASES:
        raise KeyError(
            f"Unknown metric '{metric}'. Supported: {', '.join(list_supported_metrics())}"
        )

    canonical = _ALIASES[key]
    fn = get_metric_callable(canonical)
    value = float(fn(y_true, y_proba, threshold))
    return MetricResult(name=canonical, value=value)
