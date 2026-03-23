"""Smoke tests for core Logistic Lasso via FISTA implementation.

Run:
  ./.venv/bin/python tests/smoke/test_fista_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.logistic_lasso_fista import LogisticLassoFISTA


def _nnz(coef: np.ndarray, *, tol: float = 1e-8) -> int:
    return int(np.sum(np.abs(coef) > tol))


def test_sparsity_increases_with_lambda() -> None:
    X, y = make_classification(
        n_samples=500,
        n_features=30,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=2,
        flip_y=0.03,
        class_sep=1.0,
        random_state=0,
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    lams = [0.0, 0.01, 0.05]
    nnzs = []
    objectives_end = []

    for lam in lams:
        m = LogisticLassoFISTA(max_iter=2000, tol=1e-7, backtracking=True)
        m.fit_single_lambda(Xtr, ytr, lam=lam)
        nnzs.append(_nnz(m.coef_))
        objectives_end.append(float(m.objective_history_[-1]))

        proba = m.predict_proba(Xte)
        assert np.all((proba >= 0.0) & (proba <= 1.0))
    
    # Print the results for manual inspection.
    print("Testing sparsity with increasing lambda:")
    print("Lambda values:", lams)
    print("Nonzeros in coefficients:", nnzs)
    print("Final objective values:", objectives_end)


    # As lambda increases, solution should generally become sparser.
    assert nnzs[0] >= nnzs[1] >= nnzs[2]

    # Ensure objectives are finite.
    assert all(np.isfinite(objectives_end))


def test_objective_decreases() -> None:
    X, y = make_classification(n_samples=300, n_features=20, random_state=1)
    m = LogisticLassoFISTA(max_iter=2000, tol=1e-9, backtracking=True)
    m.fit_single_lambda(X, y, lam=0.01)

    # Print the reusults to confirm
    print("Training logistic lasso with lambda=0.01")
    print("Objective values:", m.objective_history_)

    obj = m.objective_history_
    assert obj.ndim == 1
    assert len(obj) > 5
    assert float(obj[-1]) <= float(obj[0])


def test_fit_intercept_false_means_zero_intercept() -> None:
    X, y = make_classification(n_samples=200, n_features=10, random_state=2)
    m = LogisticLassoFISTA(fit_intercept=False, standardize=True, max_iter=2000)
    m.fit_single_lambda(X, y, lam=0.01)
    assert m.intercept_ == 0.0


def test_automatic_lambda_grid_descends_and_respects_ratio() -> None:
    X, y = make_classification(n_samples=300, n_features=15, random_state=3)
    m = LogisticLassoFISTA(max_iter=500, tol=1e-6)
    m.fit(X, y, n_lambdas=12, lambda_ratio=1e-2)

    assert m.lambdas_ is not None
    assert m.lambdas_.ndim == 1
    assert len(m.lambdas_) == 12
    assert np.all(m.lambdas_[:-1] >= m.lambdas_[1:])

    # If lambda_max_ is positive, check the ratio approximately matches.
    assert m.lambda_ratio_ == 1e-2
    if m.lambda_max_ is not None and m.lambda_max_ > 0:
        observed_ratio = float(m.lambdas_[-1] / m.lambdas_[0])
        assert np.isfinite(observed_ratio)
        assert abs(observed_ratio - 1e-2) < 1e-6


if __name__ == "__main__":
    test_sparsity_increases_with_lambda()
    test_objective_decreases()
    test_fit_intercept_false_means_zero_intercept()
    test_automatic_lambda_grid_descends_and_respects_ratio()
    print("fista smoke tests: OK")
