from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from src.metrics.classification import compute_metric


def _as_2d_float_array(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {arr.shape}")
    return arr


def _as_1d_int_array(y: ArrayLike) -> np.ndarray:
    arr = np.asarray(y, dtype=int)
    if arr.ndim != 1:
        arr = np.ravel(arr)
    return arr


def _check_binary_labels(y: np.ndarray) -> None:
    unique = np.unique(y)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(
            "Task 2 expects y in {0,1} with no missing labels. "
            f"Found labels: {unique.tolist()}"
        )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Stable sigmoid implementation.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _logit(p: float, *, eps: float = 1e-12) -> float:
    p_clip = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p_clip / (1.0 - p_clip)))


def _logistic_loss_mean(y: np.ndarray, z: np.ndarray) -> float:
    """Mean logistic negative log-likelihood.

    Uses the stable identity:
      log(1 + exp(z)) - y*z
    implemented via logaddexp.
    """

    # log(1+exp(z)) is stable via logaddexp(0,z)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def _logistic_grad_mean(X: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, np.ndarray]:
    """Gradient of mean logistic loss w.r.t intercept and coefficients.

    For loss L = (1/n) sum [log(1+exp(z_i)) - y_i z_i],
    with z = b + Xw:
      dL/db = mean(sigmoid(z) - y)
      dL/dw = X^T (sigmoid(z) - y) / n
    """

    p = _sigmoid(z)
    r = p - y
    grad_b = float(np.mean(r))
    grad_w = X.T @ r / X.shape[0]
    return grad_b, grad_w


def _soft_threshold(w: np.ndarray, thresh: float) -> np.ndarray:
    return np.sign(w) * np.maximum(np.abs(w) - thresh, 0.0)


def _l1_penalty(w: np.ndarray) -> float:
    return float(np.sum(np.abs(w)))


@dataclass
class Standardizer:
    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(
        cls, X: np.ndarray, *, center: bool = True, eps: float = 1e-12
    ) -> "Standardizer":
        mean = X.mean(axis=0) if center else np.zeros(X.shape[1], dtype=float)
        scale = X.std(axis=0, ddof=0)
        scale = np.where(scale < eps, 1.0, scale)
        return cls(mean_=mean, scale_=scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def inverse_coef(self, w_std: np.ndarray, b_std: float) -> tuple[np.ndarray, float]:
        """Convert coefficients from standardized-X space back to original scale."""

        w = w_std / self.scale_
        b = float(b_std - self.mean_.dot(w))
        return w, b


@dataclass
class FISTAResult:
    coef_std: np.ndarray
    intercept_std: float
    n_iter: int
    objective_history: np.ndarray


def _fista_logistic_lasso(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    *,
    fit_intercept: bool = True,
    max_iter: int = 5000,
    tol: float = 1e-6,
    initial_L: float = 1.0,
    backtracking: bool = True,
    eta: float = 2.0,
    w0: Optional[np.ndarray] = None,
    b0: float = 0.0,
) -> FISTAResult:
    """Solve logistic lasso with FISTA on a *standardized* design matrix.

    Minimizes: mean logistic loss + lam * ||w||_1

    Notes
    - L1 penalty applies only to `w` (not intercept).
    - `X` is assumed to already be standardized if desired.
    """

    if lam < 0:
        raise ValueError("lam must be non-negative")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    n_samples, n_features = X.shape
    w = np.zeros(n_features) if w0 is None else w0.astype(float, copy=True)
    b = float(b0)

    w_prev = w.copy()
    b_prev = b

    t = 1.0
    t_prev = 1.0

    # Extrapolated (y_k) variables
    y_w = w.copy()
    y_b = b

    L = float(initial_L)
    obj_hist: list[float] = []

    def smooth_f(b_i: float, w_i: np.ndarray) -> float:
        z_i = (X @ w_i) + (b_i if fit_intercept else 0.0)
        return _logistic_loss_mean(y, z_i)

    def full_obj(b_i: float, w_i: np.ndarray) -> float:
        return smooth_f(b_i, w_i) + lam * _l1_penalty(w_i)

    for k in range(1, max_iter + 1):
        # Nesterov extrapolation
        momentum = (t_prev - 1.0) / t if k > 1 else 0.0
        y_w = w + momentum * (w - w_prev)
        y_b = b + momentum * (b - b_prev)

        z_y = (X @ y_w) + (y_b if fit_intercept else 0.0)
        grad_b, grad_w = _logistic_grad_mean(X, y, z_y)

        # Backtracking line search on smooth part
        if backtracking:
            L_local = L
            f_y = _logistic_loss_mean(y, z_y)

            while True:
                step = 1.0 / L_local
                w_tmp = y_w - step * grad_w
                b_tmp = y_b - step * grad_b if fit_intercept else 0.0
                w_new = _soft_threshold(w_tmp, step * lam)
                b_new = b_tmp

                # Check majorization condition for smooth part
                z_new = (X @ w_new) + (b_new if fit_intercept else 0.0)
                f_new = _logistic_loss_mean(y, z_new)

                dw = w_new - y_w
                db = (b_new - y_b) if fit_intercept else 0.0
                quad = f_y + grad_w.dot(dw) + grad_b * db + 0.5 * L_local * (
                    float(dw.dot(dw)) + float(db * db)
                )

                if f_new <= quad + 1e-12:
                    L = L_local
                    break
                L_local *= eta
        else:
            step = 1.0 / L
            w_tmp = y_w - step * grad_w
            b_tmp = y_b - step * grad_b if fit_intercept else 0.0
            w_new = _soft_threshold(w_tmp, step * lam)
            b_new = b_tmp

        # Update t
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))

        # Bookkeeping for next iter
        w_prev, b_prev = w, b
        w, b = w_new, float(b_new)
        t_prev, t = t, t_next

        obj = full_obj(b, w)
        obj_hist.append(obj)

        if k > 5:
            # Relative change in objective
            denom = max(1.0, abs(obj_hist[-2]))
            if abs(obj_hist[-2] - obj_hist[-1]) / denom < tol:
                break

    return FISTAResult(
        coef_std=w,
        intercept_std=float(b if fit_intercept else 0.0),
        n_iter=k,
        objective_history=np.asarray(obj_hist, dtype=float),
    )


class LogisticLassoFISTA:
    """Logistic regression with an L1 penalty solved via FISTA.

    This class focuses on the core optimization (Steps 3–5). Higher-level
    validation over a lambda path and plotting will be added in later steps.
    """

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        standardize: bool = True,
        max_iter: int = 5000,
        tol: float = 1e-6,
        backtracking: bool = True,
        initial_L: float = 1.0,
        eta: float = 2.0,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.max_iter = max_iter
        self.tol = tol
        self.backtracking = backtracking
        self.initial_L = initial_L
        self.eta = eta

        self.standardizer_: Optional[Standardizer] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.objective_history_: Optional[np.ndarray] = None
        self.lam_: Optional[float] = None

        # Lambda-path / validation state (populated by fit/validate)
        self.lambdas_: Optional[np.ndarray] = None
        self.coef_path_: Optional[np.ndarray] = None  # shape (n_lambdas, n_features)
        self.intercept_path_: Optional[np.ndarray] = None  # shape (n_lambdas,)
        self.n_iter_path_: Optional[np.ndarray] = None  # shape (n_lambdas,)
        self.validation_scores_: Optional[np.ndarray] = None  # shape (n_lambdas,)
        self.metric_: Optional[str] = None
        self.threshold_: float = 0.5
        self.best_lambda_: Optional[float] = None
        self.best_index_: Optional[int] = None
        self.best_score_: Optional[float] = None

        # Lambda-grid diagnostics (Step 7)
        self.lambda_max_: Optional[float] = None
        self.lambda_min_: Optional[float] = None
        self.lambda_ratio_: Optional[float] = None
        self.n_lambdas_: Optional[int] = None

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        if self.standardize:
            if self.standardizer_ is None:
                raise RuntimeError("Standardizer is not fit")
            return self.standardizer_.transform(X)
        return X

    def _fit_standardizer(self, X_train: np.ndarray) -> np.ndarray:
        if self.standardize:
            # If we don't fit an intercept, centering X changes the hypothesis
            # class (it effectively introduces an intercept when mapping back).
            self.standardizer_ = Standardizer.fit(X_train, center=self.fit_intercept)
            return self.standardizer_.transform(X_train)
        self.standardizer_ = None
        return X_train

    def _default_lambda_grid(
        self,
        X_std: np.ndarray,
        y: np.ndarray,
        *,
        n_lambdas: int = 50,
        lambda_ratio: float = 1e-3,
    ) -> np.ndarray:
        if n_lambdas < 2:
            raise ValueError("n_lambdas must be >= 2")
        if not (0.0 < lambda_ratio < 1.0):
            raise ValueError("lambda_ratio must be in (0,1)")

        # Approximate lambda_max such that w=0 is optimal.
        # For the mean loss, the KKT condition for w=0 is:
        #   ||grad_w(w=0, b=b0)||_inf <= lambda
        # so we set:
        #   lambda_max = ||grad_w(w=0, b=b0)||_inf.
        if self.fit_intercept:
            b0 = _logit(float(np.mean(y)))
        else:
            b0 = 0.0
        z0 = b0 + X_std @ np.zeros(X_std.shape[1], dtype=float)
        _, grad_w0 = _logistic_grad_mean(X_std, y, z0)
        lam_max = float(np.max(np.abs(grad_w0)))

        if lam_max <= 0.0:
            self.lambda_max_ = float(lam_max)
            self.lambda_min_ = 0.0
            self.lambda_ratio_ = float(lambda_ratio)
            self.n_lambdas_ = int(n_lambdas)
            return np.zeros(n_lambdas, dtype=float)

        lam_min = lam_max * float(lambda_ratio)
        self.lambda_max_ = float(lam_max)
        self.lambda_min_ = float(lam_min)
        self.lambda_ratio_ = float(lambda_ratio)
        self.n_lambdas_ = int(n_lambdas)
        # Descending grid for warm starts (high -> low).
        return np.exp(np.linspace(np.log(lam_max), np.log(lam_min), n_lambdas))

    def fit_single_lambda(self, X: ArrayLike, y: ArrayLike, *, lam: float) -> "LogisticLassoFISTA":
        """Fit a single lambda value (core solver entrypoint)."""

        X_arr = _as_2d_float_array(X)
        y_arr = _as_1d_int_array(y)
        _check_binary_labels(y_arr)

        X_std = self._fit_standardizer(X_arr)

        res = _fista_logistic_lasso(
            X_std,
            y_arr,
            lam,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            initial_L=self.initial_L,
            backtracking=self.backtracking,
            eta=self.eta,
        )

        if self.standardizer_ is not None:
            w, b = self.standardizer_.inverse_coef(res.coef_std, res.intercept_std)
        else:
            w, b = res.coef_std, res.intercept_std

        if not self.fit_intercept:
            b = 0.0

        self.coef_ = w
        self.intercept_ = float(b)
        self.n_iter_ = int(res.n_iter)
        self.objective_history_ = res.objective_history
        self.lam_ = float(lam)
        return self

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        *,
        X_valid: ArrayLike | None = None,
        y_valid: ArrayLike | None = None,
        metric: str = "balanced_accuracy",
        threshold: float = 0.5,
        lambdas: ArrayLike | None = None,
        n_lambdas: int = 50,
        lambda_ratio: float = 1e-3,
    ) -> "LogisticLassoFISTA":
        """Fit along a lambda path and optionally select best lambda by validation.

        Parameters
        - X_train, y_train: training data (y in {0,1})
        - X_valid, y_valid: optional validation data
        - metric: validation metric name (see `src.metrics.classification`)
        - threshold: used for threshold-based metrics
        - lambdas: explicit lambda grid (if None, an automatic grid is created)
        - n_lambdas, lambda_ratio: controls automatic grid when lambdas is None

        Notes
        - The lambda path is fit in descending order (warm starts).
        - If validation data is provided, `validate()` is called and the model
          parameters are set to the best lambda.
        """

        Xtr = _as_2d_float_array(X_train)
        ytr = _as_1d_int_array(y_train)
        _check_binary_labels(ytr)

        self.metric_ = metric
        self.threshold_ = float(threshold)

        Xtr_std = self._fit_standardizer(Xtr)

        if lambdas is None:
            # Populate Step-7 diagnostics as a side-effect.
            lams = self._default_lambda_grid(
                Xtr_std, ytr, n_lambdas=n_lambdas, lambda_ratio=lambda_ratio
            )
        else:
            self.lambda_max_ = None
            self.lambda_min_ = None
            self.lambda_ratio_ = None
            self.n_lambdas_ = None
            lams = np.asarray(lambdas, dtype=float)
            if lams.ndim != 1:
                lams = np.ravel(lams)
            if np.any(lams < 0):
                raise ValueError("All lambdas must be non-negative")
            # Fit in descending order for warm starts.
            lams = np.sort(lams)[::-1]

        self.lambdas_ = lams

        n_features = Xtr.shape[1]
        coef_path_std = np.zeros((lams.size, n_features), dtype=float)
        intercept_path_std = np.zeros(lams.size, dtype=float)
        n_iter_path = np.zeros(lams.size, dtype=int)

        # Warm-start initial values
        w0 = np.zeros(n_features, dtype=float)
        b0 = _logit(float(np.mean(ytr))) if self.fit_intercept else 0.0

        for i, lam in enumerate(lams):
            res = _fista_logistic_lasso(
                Xtr_std,
                ytr,
                float(lam),
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                initial_L=self.initial_L,
                backtracking=self.backtracking,
                eta=self.eta,
                w0=w0,
                b0=b0,
            )
            coef_path_std[i] = res.coef_std
            intercept_path_std[i] = res.intercept_std
            n_iter_path[i] = res.n_iter

            w0 = res.coef_std
            b0 = res.intercept_std

        # Convert to original scale for storage/prediction
        if self.standardizer_ is not None:
            coef_path = np.zeros_like(coef_path_std)
            intercept_path = np.zeros_like(intercept_path_std)
            for i in range(lams.size):
                w_i, b_i = self.standardizer_.inverse_coef(coef_path_std[i], intercept_path_std[i])
                coef_path[i] = w_i
                intercept_path[i] = b_i
        else:
            coef_path = coef_path_std
            intercept_path = intercept_path_std

        if not self.fit_intercept:
            intercept_path = np.zeros_like(intercept_path)

        self.coef_path_ = coef_path
        self.intercept_path_ = intercept_path
        self.n_iter_path_ = n_iter_path

        # Default to the least-regularized model if no validation is provided.
        self.best_index_ = int(lams.size - 1)
        self.best_lambda_ = float(lams[self.best_index_])
        self.best_score_ = None

        if X_valid is not None or y_valid is not None:
            if X_valid is None or y_valid is None:
                raise ValueError("Both X_valid and y_valid must be provided")
            self.validate(X_valid, y_valid, metric=metric, threshold=threshold)

        # Set final parameters to best lambda
        idx = int(self.best_index_)
        self.coef_ = self.coef_path_[idx].copy()
        self.intercept_ = float(self.intercept_path_[idx])
        self.lam_ = float(self.lambdas_[idx])
        self.n_iter_ = int(self.n_iter_path_[idx])
        self.objective_history_ = None
        return self

    def validate(
        self,
        X_valid: ArrayLike,
        y_valid: ArrayLike,
        *,
        metric: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Compute validation scores across the fitted lambda path.

        Returns an array of scores aligned with `self.lambdas_`.
        """

        if self.lambdas_ is None or self.coef_path_ is None or self.intercept_path_ is None:
            raise RuntimeError("Must call fit() before validate()")

        Xv = _as_2d_float_array(X_valid)
        yv = _as_1d_int_array(y_valid)
        _check_binary_labels(yv)

        metric_name = (metric or self.metric_ or "balanced_accuracy")
        thr = float(self.threshold_ if threshold is None else threshold)

        scores = np.full(self.lambdas_.shape, np.nan, dtype=float)
        for i in range(self.lambdas_.size):
            proba = _sigmoid(Xv @ self.coef_path_[i] + float(self.intercept_path_[i]))
            scores[i] = compute_metric(metric_name, yv, proba, threshold=thr).value

        self.validation_scores_ = scores
        self.metric_ = metric_name
        self.threshold_ = thr

        # Select best lambda: maximize metric, ignoring NaNs.
        if np.all(np.isnan(scores)):
            raise ValueError(
                f"Validation metric '{metric_name}' is undefined for all lambdas (all NaN)."
            )
        best_idx = int(np.nanargmax(scores))
        self.best_index_ = best_idx
        self.best_lambda_ = float(self.lambdas_[best_idx])
        self.best_score_ = float(scores[best_idx])
        return scores

    def plot(self):
        """Plot validation metric vs lambda.

        Requires `fit(..., X_valid=..., y_valid=...)` or an explicit `validate()`.
        """

        if self.lambdas_ is None or self.validation_scores_ is None:
            raise RuntimeError("No validation scores to plot; run validate() first")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(self.lambdas_, self.validation_scores_, marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("lambda")
        ax.set_ylabel(self.metric_ or "metric")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        return fig, ax

    def plot_coefficients(self, *, feature_names: Optional[list[str]] = None):
        """Plot coefficient paths vs lambda.

        Requires `fit()` to have been called.
        """

        if self.lambdas_ is None or self.coef_path_ is None:
            raise RuntimeError("No coefficient path to plot; run fit() first")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        n_features = self.coef_path_.shape[1]
        names = feature_names
        if names is not None and len(names) != n_features:
            raise ValueError("feature_names length must match number of features")

        for j in range(n_features):
            label = names[j] if names is not None else None
            ax.plot(self.lambdas_, self.coef_path_[:, j], linewidth=1.0, label=label)

        ax.set_xscale("log")
        ax.set_xlabel("lambda")
        ax.set_ylabel("coefficient")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        if names is not None and n_features <= 20:
            ax.legend(loc="best", fontsize="small")
        return fig, ax

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model is not fit")
        X_arr = _as_2d_float_array(X)
        return X_arr @ self.coef_ + self.intercept_

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return P(y=1|x) as a 1D array."""

        scores = self.decision_function(X)
        return _sigmoid(scores)
