"""Task 2: more comprehensive FISTA vs sklearn comparison (multi-dataset, multi-metric, repeated splits).

This script is a "comparison suite" that:
- runs repeated train/valid/test splits (to reduce dependence on one split),
- evaluates multiple *selection metrics* (how lambda/C is chosen on validation),
- reports multiple *test metrics* for the selected model,
- compares both custom FISTA Logistic Lasso and sklearn L1 LogisticRegression on the same lambda grid,
- saves all artifacts (CSV/MD/PNG) under `--save-dir`.

Run (example):
  ./.venv/bin/python -m src.experiments.task2_comparison_suite --save-dir outputs/task2_suite

Run (full suite):
  ./.venv/bin/python -m src.experiments.task2_comparison_suite --save-dir outputs/task2_suite --datasets bioresponse.csv,spectrometer.csv,musk.csv --selection-metrics balanced_accuracy,roc_auc,pr_auc --eval-metrics balanced_accuracy,f1,roc_auc,pr_auc --n-lambdas 12 --lambda-ratio 1e-3 --n-repeats 5 --seed0 0

Notes
- All datasets are expected to be fully labeled Task-2 inputs: y in {0,1}, no missing values.
- The repo's CSVs in `data/` are already standardized; by default we set:
    custom standardize=False
    sklearn uses raw X (no StandardScaler)

Regularization mapping
- Custom objective: mean loss + lambda * ||w||_1
- sklearn uses `C` as inverse regularization strength; practical mapping:
    C ≈ 1/(lambda * n_train)

Outputs (under save-dir)
- `raw_results.csv`: one row per (dataset, seed, selection_metric, method)
- `agg_results.csv`: mean/std aggregated across seeds
- `agg_results.md`: markdown rendering of aggregated results
- `metric_vs_lambda_*.png`: validation curves for the first seed
- `test_boxplot_*.png`: boxplots over seeds for the chosen selection metric
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.metrics.classification import compute_metric
from src.models.logistic_lasso_fista import LogisticLassoFISTA


MetricList = list[str]


@dataclass(frozen=True)
class Dataset:
    name: str
    path: str


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _nnz(w: np.ndarray, *, tol: float = 1e-8) -> int:
    return int(np.sum(np.abs(w) > tol))


def _check_task2_labels(y: np.ndarray) -> None:
    unique = np.unique(y)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(f"Task 2 requires y in {{0,1}}. Found labels: {unique.tolist()}")


def _load_csv_dataset(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    if "class" not in df.columns:
        raise ValueError(f"{path}: expected a 'class' column")

    y = df["class"].to_numpy(dtype=int, copy=True)
    X_df = df.drop(columns=["class"])
    X = X_df.to_numpy(dtype=float, copy=True)

    _check_task2_labels(y)

    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError(f"{path}: contains NaNs; Task 2 requires fully observed X and y")

    return X, y, list(X_df.columns)


def _split_three_way(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    valid_frac: float,
    test_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < valid_frac < 1.0) or not (0.0 < test_frac < 1.0):
        raise ValueError("valid_frac and test_frac must be in (0,1)")
    if valid_frac + test_frac >= 1.0:
        raise ValueError("valid_frac + test_frac must be < 1")

    Xtr, Xtemp, ytr, ytemp = train_test_split(
        X, y, test_size=(valid_frac + test_frac), stratify=y, random_state=seed
    )
    rel_test = test_frac / (valid_frac + test_frac)
    Xv, Xte, yv, yte = train_test_split(
        Xtemp, ytemp, test_size=rel_test, stratify=ytemp, random_state=seed
    )
    return Xtr, Xv, Xte, ytr, yv, yte


def _lambda_to_C(lambdas: np.ndarray, n_train: int) -> np.ndarray:
    Cs = np.empty_like(lambdas, dtype=float)
    for i, lam in enumerate(lambdas):
        if lam <= 0:
            Cs[i] = 1e6
        else:
            Cs[i] = 1.0 / (float(lam) * float(n_train))
    return Cs


def _select_best(scores: np.ndarray) -> int:
    if np.all(np.isnan(scores)):
        raise ValueError("All validation scores are NaN; cannot select best hyperparameter")
    return int(np.nanargmax(scores))


def _metric_scores_over_path(
    y_true: np.ndarray,
    proba_path: np.ndarray,
    *,
    metric: str,
    threshold: float,
) -> np.ndarray:
    # proba_path: shape (n_samples, n_lambdas)
    scores = np.full(proba_path.shape[1], np.nan, dtype=float)
    for j in range(proba_path.shape[1]):
        scores[j] = float(compute_metric(metric, y_true, proba_path[:, j], threshold=threshold).value)
    return scores


def _evaluate_on_test(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    threshold: float,
    eval_metrics: MetricList,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in eval_metrics:
        out[m] = float(compute_metric(m, y_true, proba, threshold=threshold).value)
    out["accuracy"] = float(_accuracy(y_true, (proba >= threshold).astype(int)))
    return out


def _coef_similarity(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = a.astype(float, copy=False)
    b = b.astype(float, copy=False)
    l2_diff = float(np.linalg.norm(a - b))
    denom = max(1e-12, float(np.linalg.norm(a)) * float(np.linalg.norm(b)))
    cosine = float(a.dot(b) / denom)
    return {"coef_cosine": cosine, "coef_l2_diff": l2_diff}


def _fit_custom_path(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    *,
    n_lambdas: int,
    lambda_ratio: float,
    max_iter: int,
    tol: float,
    standardize: bool,
) -> LogisticLassoFISTA:
    model = LogisticLassoFISTA(
        standardize=standardize,
        max_iter=max_iter,
        tol=tol,
        backtracking=True,
    )
    model.fit(Xtr, ytr, n_lambdas=n_lambdas, lambda_ratio=lambda_ratio)
    return model


def _fit_sklearn_path(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    *,
    Cs: np.ndarray,
    solver: str,
    max_iter: int,
    tol: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Returns (coef_path, intercept_path)
    warm_start = solver == "saga"
    lr = LogisticRegression(
        penalty="l1",
        solver=solver,
        C=float(Cs[0]),
        max_iter=int(max_iter),
        tol=float(tol),
        fit_intercept=True,
        random_state=int(random_state),
        warm_start=warm_start,
    )

    coef_path = np.zeros((Cs.size, Xtr.shape[1]), dtype=float)
    intercept_path = np.zeros(Cs.size, dtype=float)

    for i, C in enumerate(Cs):
        lr.set_params(C=float(C))
        lr.fit(Xtr, ytr)
        coef_path[i] = lr.coef_.ravel().astype(float)
        intercept_path[i] = float(lr.intercept_.ravel()[0])

    return coef_path, intercept_path


def _plot_metric_vs_lambda(
    save_path: Path,
    lambdas: np.ndarray,
    custom_scores: np.ndarray,
    sk_scores: np.ndarray,
    *,
    metric: str,
    dataset_name: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(lambdas, custom_scores, marker="o", label="custom (FISTA)")
    ax.plot(lambdas, sk_scores, marker="x", label="sklearn (mapped grid)")
    ax.set_xscale("log")
    ax.set_xlabel("lambda")
    ax.set_ylabel(metric)
    ax.set_title(f"{dataset_name}: validation {metric} vs lambda")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(loc="best")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_nnz_vs_lambda(
    save_path: Path,
    lambdas: np.ndarray,
    nnz_custom: np.ndarray,
    nnz_sklearn: np.ndarray,
    *,
    dataset_name: str,
    sklearn_solver: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(lambdas, nnz_custom, marker="o", label="custom (FISTA) nnz")
    ax.plot(lambdas, nnz_sklearn, marker="x", label=f"sklearn ({sklearn_solver}) nnz")
    ax.set_xscale("log")
    ax.set_xlabel("lambda")
    ax.set_ylabel("# non-zero coefficients")
    ax.set_title(f"{dataset_name}: sparsity path (nnz) vs lambda")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(loc="best")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_boxplot(
    save_path: Path,
    values_custom: Iterable[float],
    values_sklearn: Iterable[float],
    *,
    title: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.boxplot(
        [list(values_custom), list(values_sklearn)],
        tick_labels=["custom", "sklearn"],
        showmeans=True,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"

    cols = list(df.columns)
    rows = [cols] + df.astype(str).values.tolist()
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(cols))]

    def fmt_row(r: list[str]) -> str:
        return "| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(cols))) + " |"

    header = fmt_row(cols)
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = "\n".join(fmt_row(r) for r in df.astype(str).values.tolist())
    return "\n".join([header, sep, body])


def _latex_escape(s: str) -> str:
    return s.replace("\\", "\\textbackslash{}").replace("_", "\\_")


def _render_latex_table(df: pd.DataFrame, *, caption: str, label: str) -> str:
    """Render a simple booktabs LaTeX table without optional pandas dependencies."""

    if df.empty:
        return "% (empty table)"

    cols = list(df.columns)
    # l for first three identifier columns, r for numeric columns
    colspec = "".join(["l" if i < 3 else "r" for i in range(len(cols))])

    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    lines.append(f"\\label{{{_latex_escape(label)}}}")
    lines.append(f"\\begin{{tabular}}{{@{{}}{colspec}@{{}}}}")
    lines.append("\\toprule")

    header = " & ".join(_latex_escape(str(c)) for c in cols) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for _, row in df.iterrows():
        items: list[str] = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                items.append(f"{float(v):.4f}")
            elif isinstance(v, (int, np.integer)):
                items.append(str(int(v)))
            else:
                items.append(_latex_escape(str(v)))
        lines.append(" & ".join(items) + " \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="bioresponse.csv,spectrometer.csv",
        help="Comma-separated dataset CSV filenames under data/.",
    )
    parser.add_argument(
        "--selection-metrics",
        type=str,
        default="balanced_accuracy,roc_auc,pr_auc",
        help="Comma-separated list of metrics used to pick best lambda on validation.",
    )
    parser.add_argument(
        "--eval-metrics",
        type=str,
        default="balanced_accuracy,f1,roc_auc,pr_auc",
        help="Comma-separated list of metrics to report on the test set (plus accuracy).",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-lambdas", type=int, default=12)
    parser.add_argument("--lambda-ratio", type=float, default=1e-3)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--valid-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument(
        "--custom-standardize",
        action="store_true",
        help="If set, standardize features inside the custom model (usually unnecessary for pre-standardized CSVs).",
    )
    parser.add_argument("--custom-max-iter", type=int, default=5000)
    parser.add_argument("--custom-tol", type=float, default=1e-7)
    parser.add_argument(
        "--sklearn-solver",
        type=str,
        default="liblinear",
        choices=["liblinear", "saga"],
        help="Solver for sklearn LogisticRegression.",
    )
    parser.add_argument("--sklearn-max-iter", type=int, default=2000)
    parser.add_argument("--sklearn-tol", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    save_dir = _ensure_dir(args.save_dir)
    data_dir = Path("data")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    selection_metrics = [m.strip() for m in args.selection_metrics.split(",") if m.strip()]
    eval_metrics = [m.strip() for m in args.eval_metrics.split(",") if m.strip()]

    seeds = [int(args.seed0) + i for i in range(int(args.n_repeats))]

    rows: list[dict[str, object]] = []

    # We also keep the first repeat's validation curves for plotting.
    first_repeat_curves: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    first_repeat_nnz: dict[str, dict[str, np.ndarray]] = {}

    for dataset_fn in datasets:
        ds = Dataset(name=dataset_fn.replace(".csv", ""), path=str(data_dir / dataset_fn))
        X, y, _feature_names = _load_csv_dataset(ds.path)

        for rep_idx, seed in enumerate(seeds):
            Xtr, Xv, Xte, ytr, yv, yte = _split_three_way(
                X, y, seed=seed, valid_frac=float(args.valid_frac), test_frac=float(args.test_frac)
            )

            custom = _fit_custom_path(
                Xtr,
                ytr,
                n_lambdas=int(args.n_lambdas),
                lambda_ratio=float(args.lambda_ratio),
                max_iter=int(args.custom_max_iter),
                tol=float(args.custom_tol),
                standardize=bool(args.custom_standardize),
            )
            assert custom.lambdas_ is not None and custom.coef_path_ is not None and custom.intercept_path_ is not None
            lambdas = custom.lambdas_.astype(float)

            # Compute probabilities over path for valid and test.
            Zv = Xv @ custom.coef_path_.T + custom.intercept_path_[None, :]
            Zt = Xte @ custom.coef_path_.T + custom.intercept_path_[None, :]
            custom_proba_valid = _sigmoid(Zv)
            custom_proba_test = _sigmoid(Zt)

            Cs = _lambda_to_C(lambdas, n_train=Xtr.shape[0])
            sk_coef_path, sk_intercept_path = _fit_sklearn_path(
                Xtr,
                ytr,
                Cs=Cs,
                solver=str(args.sklearn_solver),
                max_iter=int(args.sklearn_max_iter),
                tol=float(args.sklearn_tol),
                random_state=int(seed),
            )
            Zsv = Xv @ sk_coef_path.T + sk_intercept_path[None, :]
            Zst = Xte @ sk_coef_path.T + sk_intercept_path[None, :]
            sk_proba_valid = _sigmoid(Zsv)
            sk_proba_test = _sigmoid(Zst)

            if rep_idx == 0 and not args.no_plots:
                # Save sparsity path once per dataset (first repeat only)
                nnz_custom = np.array([_nnz(w) for w in custom.coef_path_], dtype=int)
                nnz_sklearn = np.array([_nnz(w) for w in sk_coef_path], dtype=int)
                first_repeat_nnz[ds.name] = {
                    "lambdas": lambdas,
                    "custom": nnz_custom,
                    "sklearn": nnz_sklearn,
                }

            for sel_metric in selection_metrics:
                # Validation score paths
                custom_valid_scores = _metric_scores_over_path(
                    yv, custom_proba_valid, metric=sel_metric, threshold=float(args.threshold)
                )
                sk_valid_scores = _metric_scores_over_path(
                    yv, sk_proba_valid, metric=sel_metric, threshold=float(args.threshold)
                )

                if rep_idx == 0 and not args.no_plots:
                    first_repeat_curves[(ds.name, sel_metric)] = {
                        "lambdas": lambdas,
                        "custom": custom_valid_scores,
                        "sklearn": sk_valid_scores,
                    }

                # Select best index separately per method
                best_idx_custom = _select_best(custom_valid_scores)
                best_idx_sk = _select_best(sk_valid_scores)

                # Evaluate on test at selected indices
                custom_test = _evaluate_on_test(
                    yte,
                    custom_proba_test[:, best_idx_custom],
                    threshold=float(args.threshold),
                    eval_metrics=eval_metrics,
                )
                sk_test = _evaluate_on_test(
                    yte,
                    sk_proba_test[:, best_idx_sk],
                    threshold=float(args.threshold),
                    eval_metrics=eval_metrics,
                )

                # Compare coefficients at the *custom-selected* lambda index (same lambda grid)
                sim = _coef_similarity(custom.coef_path_[best_idx_custom], sk_coef_path[best_idx_custom])

                rows.append(
                    {
                        "dataset": ds.name,
                        "seed": int(seed),
                        "selection_metric": sel_metric,
                        "method": "custom_fista",
                        "best_index": int(best_idx_custom),
                        "best_lambda": float(lambdas[best_idx_custom]),
                        "mapped_C": float(Cs[best_idx_custom]),
                        "valid_score": float(custom_valid_scores[best_idx_custom]),
                        "nnz": int(_nnz(custom.coef_path_[best_idx_custom])),
                        **{f"test_{k}": float(v) for k, v in custom_test.items()},
                        **sim,
                    }
                )

                rows.append(
                    {
                        "dataset": ds.name,
                        "seed": int(seed),
                        "selection_metric": sel_metric,
                        "method": f"sklearn_l1_{args.sklearn_solver}",
                        "best_index": int(best_idx_sk),
                        "best_lambda": float(lambdas[best_idx_sk]),
                        "mapped_C": float(Cs[best_idx_sk]),
                        "valid_score": float(sk_valid_scores[best_idx_sk]),
                        "nnz": int(_nnz(sk_coef_path[best_idx_sk])),
                        **{f"test_{k}": float(v) for k, v in sk_test.items()},
                        **sim,
                    }
                )

    raw = pd.DataFrame(rows)
    raw_path = save_dir / "raw_results.csv"
    raw.to_csv(raw_path, index=False)

    # Aggregation: mean and std across seeds.
    metric_cols = [c for c in raw.columns if c.startswith("test_") or c in {"valid_score", "nnz", "coef_cosine", "coef_l2_diff"}]
    group_cols = ["dataset", "selection_metric", "method"]

    agg_mean = raw.groupby(group_cols)[metric_cols].mean(numeric_only=True).reset_index()
    agg_std = raw.groupby(group_cols)[metric_cols].std(ddof=0, numeric_only=True).reset_index()

    # Flatten into mean/std columns
    agg = agg_mean[group_cols].copy()
    for c in metric_cols:
        agg[f"{c}_mean"] = agg_mean[c]
        agg[f"{c}_std"] = agg_std[c]

    agg_path = save_dir / "agg_results.csv"
    agg.to_csv(agg_path, index=False)

    # Markdown summary
    md_lines = []
    md_lines.append("# Task 2 — FISTA vs scikit-learn comparison suite")
    md_lines.append("")
    md_lines.append(f"- datasets: {datasets}")
    md_lines.append(f"- selection metrics: {selection_metrics}")
    md_lines.append(f"- repeats: {len(seeds)} (seeds={seeds})")
    md_lines.append(f"- lambda grid: n_lambdas={args.n_lambdas}, lambda_ratio={args.lambda_ratio}")
    md_lines.append(f"- sklearn solver: {args.sklearn_solver}")
    md_lines.append("")

    focus_cols = [
        "dataset",
        "selection_metric",
        "method",
        "valid_score_mean",
        "valid_score_std",
        "test_balanced_accuracy_mean",
        "test_balanced_accuracy_std",
        "test_roc_auc_mean",
        "test_roc_auc_std",
        "test_pr_auc_mean",
        "test_pr_auc_std",
        "test_accuracy_mean",
        "test_accuracy_std",
        "nnz_mean",
        "nnz_std",
    ]
    focus_cols = [c for c in focus_cols if c in agg.columns]
    md_lines.append("## Aggregated results (mean ± std)")
    md_lines.append("")
    md_lines.append(_render_markdown_table(agg[focus_cols].round(6)))
    md_lines.append("")
    md_lines.append("Raw results: raw_results.csv")
    md_lines.append("Aggregated results: agg_results.csv")

    md_path = save_dir / "agg_results.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    tex_path = save_dir / "agg_results_focus.tex"
    tex_path.write_text(
        _render_latex_table(
            agg[focus_cols],
            caption="Task 2: aggregated comparison (mean/std across repeats)",
            label="tab:task2-fista-vs-sklearn",
        ),
        encoding="utf-8",
    )

    # Plots from first repeat
    if not args.no_plots:
        for (ds_name, sel_metric), curves in first_repeat_curves.items():
            out = save_dir / f"metric_vs_lambda_{ds_name}_{sel_metric}.png"
            _plot_metric_vs_lambda(
                out,
                curves["lambdas"],
                curves["custom"],
                curves["sklearn"],
                metric=sel_metric,
                dataset_name=ds_name,
            )

        for ds_name, nnz in first_repeat_nnz.items():
            out = save_dir / f"nnz_vs_lambda_{ds_name}.png"
            _plot_nnz_vs_lambda(
                out,
                nnz["lambdas"],
                nnz["custom"],
                nnz["sklearn"],
                dataset_name=ds_name,
                sklearn_solver=str(args.sklearn_solver),
            )

        # Boxplots over seeds for the *test value of the selection metric*.
        for ds_name in sorted(set(raw["dataset"])):
            for sel_metric in selection_metrics:
                sub = raw[(raw["dataset"] == ds_name) & (raw["selection_metric"] == sel_metric)]
                if sub.empty:
                    continue
                custom_vals = sub[sub["method"] == "custom_fista"][f"test_{sel_metric}"]
                sk_vals = sub[sub["method"].str.startswith("sklearn_l1_")][f"test_{sel_metric}"]
                if custom_vals.empty or sk_vals.empty:
                    continue

                out = save_dir / f"test_boxplot_{ds_name}_{sel_metric}.png"
                _plot_boxplot(
                    out,
                    custom_vals,
                    sk_vals,
                    title=f"{ds_name}: test {sel_metric} over repeats (selected by {sel_metric})",
                    ylabel=sel_metric,
                )

    print(f"Saved: {raw_path}")
    print(f"Saved: {agg_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {tex_path}")


if __name__ == "__main__":
    main()
