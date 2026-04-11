"""Compare custom FISTA Logistic Lasso vs scikit-learn L1 LogisticRegression on bioresponse.csv.

This is a Task 2 (fully-labeled) evaluation on a real dataset included in the repo.

It mirrors `task2_sklearn_comparison.py`, except the dataset is loaded from CSV.

Run (example):
  ./.venv/bin/python -m src.experiments.task2_bioresponse_comparison --no-show --save-dir outputs/bioresponse

Notes on lambda vs C mapping
- Our objective uses mean loss + lambda * ||w||_1.
- sklearn's `LogisticRegression` uses `C` as inverse regularization strength,
  and its internal scaling is closer to a sum over samples.
- A common practical alignment is:
    C ≈ 1 / (lambda * n_train)

Dataset notes
- `data/bioresponse.csv` is expected to be preprocessed with:
  - target column named `class` in {0,1}
  - no missing values in X or y
  - features already standardized

Because the CSV is already standardized, this script defaults to:
- custom: standardize=False
- sklearn: no StandardScaler pipeline
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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


def _metric_table(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    threshold: float,
    metrics: list[str],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in metrics:
        out[m] = float(compute_metric(m, y_true, proba, threshold=threshold).value)
    out["accuracy"] = float(_accuracy(y_true, (proba >= threshold).astype(int)))
    return out


def _format_metric_table(values: dict[str, float]) -> str:
    parts = []
    for k, v in values.items():
        if np.isnan(v):
            parts.append(f"{k}=nan")
        else:
            parts.append(f"{k}={v:.4f}")
    return " ".join(parts)


def _objective_diagnostics(obj: np.ndarray) -> dict[str, float]:
    if obj.size < 2:
        return {
            "n_iter": float(obj.size),
            "final_obj": float(obj[-1]) if obj.size else float("nan"),
            "n_increases": 0.0,
            "max_increase": 0.0,
            "max_rel_increase": 0.0,
        }
    diffs = np.diff(obj)
    inc = diffs > 0
    max_inc = float(np.max(diffs[inc])) if np.any(inc) else 0.0
    prev = obj[:-1]
    rel = np.zeros_like(diffs)
    denom = np.maximum(1.0, np.abs(prev))
    rel[inc] = diffs[inc] / denom[inc]
    max_rel = float(np.max(rel)) if np.any(inc) else 0.0
    return {
        "n_iter": float(obj.size),
        "final_obj": float(obj[-1]),
        "n_increases": float(np.sum(inc)),
        "max_increase": max_inc,
        "max_rel_increase": max_rel,
    }


def _coef_similarity(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = a.astype(float, copy=False)
    b = b.astype(float, copy=False)
    l2_a = float(np.linalg.norm(a))
    l2_b = float(np.linalg.norm(b))
    l2_diff = float(np.linalg.norm(a - b))
    denom = max(1e-12, l2_a * l2_b)
    cos = float(a.dot(b) / denom)
    return {
        "l2_custom": l2_a,
        "l2_sklearn": l2_b,
        "l2_diff": l2_diff,
        "cosine": cos,
    }


def _topk_feature_indices_by_path_magnitude(coef_path: np.ndarray, k: int) -> np.ndarray:
    """Pick indices of the k features with the largest max |coef| along the path."""

    if k <= 0:
        return np.array([], dtype=int)
    mag = np.max(np.abs(coef_path), axis=0)
    k_eff = int(min(k, mag.size))
    # argpartition is O(p); then sort the top-k for stable plotting.
    idx = np.argpartition(-mag, kth=k_eff - 1)[:k_eff]
    idx = idx[np.argsort(-mag[idx])]
    return idx.astype(int)


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
    solver: str,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Fit sklearn models over a mapped C-grid; return valid scores and coef path.

    Returns: (C_grid, scores, best_index, coef_path, intercept_path)
    """

    n_train = Xtr.shape[0]

    Cs = np.empty_like(lambdas, dtype=float)
    for i, lam in enumerate(lambdas):
        if lam <= 0:
            Cs[i] = 1e6
        else:
            Cs[i] = 1.0 / (float(lam) * float(n_train))

    scores = np.full(Cs.shape, np.nan, dtype=float)
    coef_path = np.zeros((Cs.size, Xtr.shape[1]), dtype=float)
    intercept_path = np.zeros(Cs.size, dtype=float)

    # For solver='saga', warm_start provides a large speed-up along a regularization path.
    warm_start = solver == "saga"
    lr = LogisticRegression(
        penalty="l1",
        solver=solver,
        C=float(Cs[0]),
        max_iter=max_iter,
        tol=float(tol),
        fit_intercept=True,
        random_state=random_state,
        warm_start=warm_start,
    )

    for i, C in enumerate(Cs):
        lr.set_params(C=float(C))
        lr.fit(Xtr, ytr)
        proba = lr.predict_proba(Xv)[:, 1]
        scores[i] = compute_metric(metric, yv, proba, threshold=threshold).value

        coef_path[i] = lr.coef_.ravel().astype(float)
        intercept_path[i] = float(lr.intercept_.ravel()[0])

    if np.all(np.isnan(scores)):
        raise ValueError(f"Validation metric '{metric}' undefined for all sklearn runs")

    best_idx = int(np.nanargmax(scores))
    return Cs, scores, best_idx, coef_path, intercept_path


def _load_bioresponse(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    if "class" not in df.columns:
        raise ValueError("Expected a 'class' column in the dataset")

    y = df["class"].to_numpy(dtype=int, copy=True)
    X_df = df.drop(columns=["class"])
    X = X_df.to_numpy(dtype=float, copy=True)

    unique = np.unique(y)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(
            "Task 2 requires fully observed binary labels in {0,1}. "
            f"Found labels: {unique.tolist()}"
        )
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("Dataset contains NaNs; Task 2 requires fully observed data")

    feature_names = list(X_df.columns)
    return X, y, feature_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/bioresponse.csv")
    parser.add_argument("--metric", type=str, default="balanced_accuracy")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-lambdas", type=int, default=30)
    parser.add_argument("--lambda-ratio", type=float, default=1e-3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--sklearn-solver",
        type=str,
        default="liblinear",
        choices=["liblinear", "saga"],
        help="Solver for sklearn LogisticRegression (liblinear is often faster for this dataset).",
    )
    parser.add_argument("--sklearn-max-iter", type=int, default=2000)
    parser.add_argument("--sklearn-tol", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--report-metrics",
        type=str,
        default="balanced_accuracy,f1,roc_auc,pr_auc",
        help="Comma-separated list of metrics to print on the test set (plus accuracy).",
    )
    parser.add_argument(
        "--coef-top-k",
        type=int,
        default=30,
        help="Plot only top-k coefficient paths (by max |coef|).",
    )
    args = parser.parse_args()

    save_dir = _ensure_dir(args.save_dir)
    report_metrics = [m.strip() for m in args.report_metrics.split(",") if m.strip()]

    X, y, feature_names = _load_bioresponse(args.data_path)

    # Train/valid/test split: 60/20/20
    Xtr, Xtemp, ytr, ytemp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=args.random_state
    )
    Xv, Xte, yv, yte = train_test_split(
        Xtemp, ytemp, test_size=0.5, stratify=ytemp, random_state=args.random_state
    )

    print("=== Task 2 comparison on bioresponse.csv ===")
    print(f"Data: {args.data_path}")
    print(f"Shape: n={X.shape[0]} p={X.shape[1]} pos_rate={float(np.mean(y)):.4f}")
    print(f"Split sizes: train={Xtr.shape[0]} valid={Xv.shape[0]} test={Xte.shape[0]}")
    print(f"Metric: {args.metric} (threshold={args.threshold})")

    # --- Custom model ---
    custom = LogisticLassoFISTA(standardize=False, max_iter=5000, tol=1e-7)
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
    Cs, sk_scores, sk_best, sk_coef_path, sk_intercept_path = fit_sklearn_path(
        Xtr,
        ytr,
        Xv,
        yv,
        metric=args.metric,
        threshold=args.threshold,
        lambdas=custom.lambdas_,
        random_state=args.random_state,
        max_iter=int(args.sklearn_max_iter),
        solver=str(args.sklearn_solver),
        tol=float(args.sklearn_tol),
    )

    # Fit best sklearn model on train only (to match selection) and evaluate on test
    best_lr = LogisticRegression(
        penalty="l1",
        solver=str(args.sklearn_solver),
        C=float(Cs[sk_best]),
        max_iter=int(args.sklearn_max_iter),
        tol=float(args.sklearn_tol),
        fit_intercept=True,
        random_state=args.random_state,
    )
    best_lr.fit(Xtr, ytr)
    sk_proba = best_lr.predict_proba(Xte)[:, 1]
    sk_pred = (sk_proba >= args.threshold).astype(int)
    sk_acc = _accuracy(yte, sk_pred)
    sk_metric = compute_metric(args.metric, yte, sk_proba, threshold=args.threshold).value
    sk_nnz = _nnz(best_lr.coef_.ravel())

    sk_summary = Summary(
        name=f"sklearn_l1_{args.sklearn_solver}",
        best_param=float(Cs[sk_best]),
        best_valid_score=float(sk_scores[sk_best]),
        test_accuracy=float(sk_acc),
        test_metric=float(sk_metric),
        nnz=int(sk_nnz),
    )

    print()
    for s in [custom_summary, sk_summary]:
        print(
            f"{s.name}: best_param={s.best_param:.6g} valid={s.best_valid_score:.4f} "
            f"test_acc={s.test_accuracy:.4f} test_metric={s.test_metric:.4f} nnz={s.nnz}"
        )

    print()
    custom_test_metrics = _metric_table(yte, custom_proba, threshold=args.threshold, metrics=report_metrics)
    sk_test_metrics = _metric_table(yte, sk_proba, threshold=args.threshold, metrics=report_metrics)
    print("Test metrics (custom):  ", _format_metric_table(custom_test_metrics))
    print("Test metrics (sklearn): ", _format_metric_table(sk_test_metrics))

    # Convergence diagnostics: re-solve best lambda to get objective history
    best_lam = float(custom.best_lambda_ if custom.best_lambda_ is not None else custom.lam_)
    conv_model = LogisticLassoFISTA(
        fit_intercept=custom.fit_intercept,
        standardize=custom.standardize,
        max_iter=custom.max_iter,
        tol=custom.tol,
        backtracking=custom.backtracking,
        initial_L=custom.initial_L,
        eta=custom.eta,
    )
    conv_model.fit_single_lambda(Xtr, ytr, lam=best_lam)
    obj = conv_model.objective_history_
    if obj is not None:
        diag = _objective_diagnostics(obj)
        print()
        print(
            "Convergence (custom best lambda): "
            f"n_iter={int(diag['n_iter'])} final_obj={diag['final_obj']:.6g} "
            f"n_increases={int(diag['n_increases'])} max_rel_increase={diag['max_rel_increase']:.3g}"
        )

    # Coefficient similarity (both are on the CSV feature scale)
    assert custom.coef_ is not None
    sim = _coef_similarity(custom.coef_, best_lr.coef_.ravel().astype(float))
    print()
    print(
        "Coefficient comparison (bioresponse CSV scale): "
        f"cosine={sim['cosine']:.4f} l2_diff={sim['l2_diff']:.4f} "
        f"|b_custom-b_sklearn|={abs(float(custom.intercept_) - float(best_lr.intercept_.ravel()[0])):.4f}"
    )

    n_train = Xtr.shape[0]
    lam_from_C = 1.0 / (float(sk_summary.best_param) * float(n_train))
    print()
    print("Conventions:")
    print(f"- custom: mean loss + lambda*||w||_1, standardize={custom.standardize}, fit_intercept={custom.fit_intercept}")
    print(f"- sklearn: LogisticRegression(penalty='l1', solver='{args.sklearn_solver}') on same X scale")
    print(f"- mapping used for grid: C ≈ 1/(lambda*n_train), n_train={n_train}")
    print(f"  best lambda (custom)={best_lam:.6g}, best C (sklearn)={sk_summary.best_param:.6g} -> implied lambda≈{lam_from_C:.6g}")

    # --- Plots ---
    if args.no_show or save_dir is not None:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Metric vs lambda
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

    # Coefficient paths (top-k)
    assert custom.coef_path_ is not None
    top_idx = _topk_feature_indices_by_path_magnitude(custom.coef_path_, int(args.coef_top_k))
    fig2, ax2 = plt.subplots()
    for j in top_idx:
        label = feature_names[j] if len(top_idx) <= 20 else None
        ax2.plot(custom.lambdas_, custom.coef_path_[:, j], linewidth=1.0, label=label)
    ax2.set_xscale("log")
    ax2.set_xlabel("lambda (custom grid)")
    ax2.set_ylabel("coefficient")
    ax2.grid(True, which="both", linestyle=":", linewidth=0.5)
    if len(top_idx) <= 20:
        ax2.legend(loc="best", fontsize="small")
    ax2.set_title(f"Custom coefficient paths (top {len(top_idx)})")
    if save_dir is not None:
        fig2.savefig(save_dir / "custom_coef_paths.png", dpi=150, bbox_inches="tight")

    # Sparsity path (nnz) comparison
    fig3, ax3 = plt.subplots()
    custom_nnz_path = np.array([_nnz(w) for w in custom.coef_path_], dtype=int)
    sk_nnz_path = np.array([_nnz(w) for w in sk_coef_path], dtype=int)
    ax3.plot(custom.lambdas_, custom_nnz_path, marker="o", label="custom nnz")
    ax3.plot(custom.lambdas_, sk_nnz_path, marker="x", label="sklearn nnz")
    ax3.set_xscale("log")
    ax3.set_xlabel("lambda (custom grid)")
    ax3.set_ylabel("# non-zero coefficients")
    ax3.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax3.legend(loc="best")
    if save_dir is not None:
        fig3.savefig(save_dir / "nnz_vs_lambda.png", dpi=150, bbox_inches="tight")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
