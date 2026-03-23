"""Step 8: Compare custom FISTA Logistic Lasso vs scikit-learn L1 LogisticRegression.

This script is meant as a runnable, reproducible comparison for the Task 2 report.
It:
- creates a (fully-observed) binary classification dataset
- splits into train/valid/test
- fits `LogisticLassoFISTA` over a lambda grid and selects best lambda by a
  chosen validation metric
- fits an sklearn L1 logistic baseline over a comparable grid and selects best
  hyperparameter by the same metric
- prints performance + sparsity summaries

Run:
  ./.venv/bin/python -m src.experiments.task2_sklearn_comparison --metric roc_auc

Notes on lambda vs C mapping
- Our objective uses mean loss + lambda * ||w||_1.
- sklearn's `LogisticRegression` uses `C` as inverse regularization strength,
  and its loss is scaled differently (effectively a sum over samples).
- A common practical alignment is:
    C ≈ 1 / (lambda * n_train)
  (this is an approximation; solver/convention differences mean results won't
   match exactly).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.metrics.classification import compute_metric
from src.models.logistic_lasso_fista import LogisticLassoFISTA


@dataclass(frozen=True)
class Summary:
    name: str
    best_param: float
    best_valid_score: float
    test_accuracy: float
    test_metric: float
    nnz: int


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _nnz(w: np.ndarray, *, tol: float = 1e-8) -> int:
    return int(np.sum(np.abs(w) > tol))


def _ensure_dir(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def fit_sklearn_path(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xv: np.ndarray,
    yv: np.ndarray,
    *,
    metric: str,
    threshold: float,
    lambdas: np.ndarray,
    random_state: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit sklearn models over a mapped C-grid; return valid scores.

    Returns: (C_grid, scores, best_index)
    """

    n_train = Xtr.shape[0]

    # Map lambdas -> C. Handle lambda=0 by using a large C.
    Cs = np.empty_like(lambdas, dtype=float)
    for i, lam in enumerate(lambdas):
        if lam <= 0:
            Cs[i] = 1e6
        else:
            Cs[i] = 1.0 / (float(lam) * float(n_train))

    scores = np.full(Cs.shape, np.nan, dtype=float)

    for i, C in enumerate(Cs):
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        penalty="l1",
                        solver="saga",
                        C=float(C),
                        max_iter=max_iter,
                        tol=1e-4,
                        fit_intercept=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xv)[:, 1]
        scores[i] = compute_metric(metric, yv, proba, threshold=threshold).value

    if np.all(np.isnan(scores)):
        raise ValueError(f"Validation metric '{metric}' undefined for all sklearn runs")

    best_idx = int(np.nanargmax(scores))
    return Cs, scores, best_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="balanced_accuracy")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-lambdas", type=int, default=30)
    parser.add_argument("--lambda-ratio", type=float, default=1e-3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    save_dir = _ensure_dir(args.save_dir)

    X, y = make_classification(
        n_samples=800,
        n_features=40,
        n_informative=12,
        n_redundant=4,
        n_clusters_per_class=2,
        flip_y=0.03,
        class_sep=1.0,
        random_state=args.random_state,
    )

    Xtr, Xtemp, ytr, ytemp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=args.random_state
    )
    Xv, Xte, yv, yte = train_test_split(
        Xtemp, ytemp, test_size=0.5, stratify=ytemp, random_state=args.random_state
    )

    # --- Custom model ---
    custom = LogisticLassoFISTA(max_iter=5000, tol=1e-7)
    custom.fit(
        Xtr,
        ytr,
        X_valid=Xv,
        y_valid=yv,
        metric=args.metric,
        threshold=args.threshold,
        n_lambdas=args.n_lambdas,
        lambda_ratio=args.lambda_ratio,
    )

    custom_proba = custom.predict_proba(Xte)
    custom_pred = (custom_proba >= args.threshold).astype(int)
    custom_acc = _accuracy(yte, custom_pred)
    custom_metric = compute_metric(args.metric, yte, custom_proba, threshold=args.threshold).value
    custom_nnz = _nnz(custom.coef_)

    custom_summary = Summary(
        name="custom_fista",
        best_param=float(custom.best_lambda_ if custom.best_lambda_ is not None else custom.lam_),
        best_valid_score=float(custom.best_score_ if custom.best_score_ is not None else np.nan),
        test_accuracy=float(custom_acc),
        test_metric=float(custom_metric),
        nnz=int(custom_nnz),
    )

    # --- sklearn baseline ---
    assert custom.lambdas_ is not None
    Cs, sk_scores, sk_best = fit_sklearn_path(
        Xtr,
        ytr,
        Xv,
        yv,
        metric=args.metric,
        threshold=args.threshold,
        lambdas=custom.lambdas_,
        random_state=args.random_state,
        max_iter=5000,
    )

    # Fit best sklearn model on train only (to match selection) and evaluate on test
    best_clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=float(Cs[sk_best]),
                    max_iter=5000,
                    tol=1e-4,
                    fit_intercept=True,
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    best_clf.fit(Xtr, ytr)
    sk_proba = best_clf.predict_proba(Xte)[:, 1]
    sk_pred = (sk_proba >= args.threshold).astype(int)
    sk_acc = _accuracy(yte, sk_pred)
    sk_metric = compute_metric(args.metric, yte, sk_proba, threshold=args.threshold).value

    lr = best_clf.named_steps["lr"]
    sk_nnz = _nnz(lr.coef_.ravel())

    sk_summary = Summary(
        name="sklearn_l1_saga",
        best_param=float(Cs[sk_best]),
        best_valid_score=float(sk_scores[sk_best]),
        test_accuracy=float(sk_acc),
        test_metric=float(sk_metric),
        nnz=int(sk_nnz),
    )

    # --- Reporting ---
    print("=== Step 8 comparison ===")
    print(f"Metric: {args.metric} (threshold={args.threshold})")
    print()
    for s in [custom_summary, sk_summary]:
        print(
            f"{s.name}: best_param={s.best_param:.6g} valid={s.best_valid_score:.4f} "
            f"test_acc={s.test_accuracy:.4f} test_metric={s.test_metric:.4f} nnz={s.nnz}"
        )

    # --- Plots ---
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots()
    assert custom.validation_scores_ is not None
    ax1.plot(custom.lambdas_, custom.validation_scores_, marker="o", label="custom (lambda)")
    ax1.plot(custom.lambdas_, sk_scores, marker="x", label="sklearn (mapped grid)")
    ax1.set_xscale("log")
    ax1.set_xlabel("lambda (custom grid)")
    ax1.set_ylabel(args.metric)
    ax1.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax1.legend(loc="best")

    if save_dir is not None:
        fig1.savefig(save_dir / "metric_vs_lambda.png", dpi=150, bbox_inches="tight")

    fig2, ax2 = custom.plot_coefficients()
    if save_dir is not None:
        fig2.savefig(save_dir / "custom_coef_paths.png", dpi=150, bbox_inches="tight")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
