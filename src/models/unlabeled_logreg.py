from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.models.logistic_lasso_fista import LogisticLassoFISTA, _as_2d_float_array, _as_1d_int_array

CompletionMethod = Literal["mean", "knn_mean", "mean_plus_S"]


def _labeled_mask(y_obs: np.ndarray) -> np.ndarray:
    return y_obs >= 0


def _complete_y_mean(X: np.ndarray, y_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Impute missing labels with majority class from sample mean of observed labels."""

    labeled = _labeled_mask(y_obs)
    if not np.any(labeled):
        raise ValueError("No labeled observations in y_obs.")
    p = float(np.mean(y_obs[labeled]))
    y_full = y_obs.copy()
    y_full[~labeled] = 1 if p >= 0.5 else 0
    return X, y_full.astype(int)


def _complete_y_knn_mean(
    X: np.ndarray, y_obs: np.ndarray, *, knn_k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Average label among k nearest labeled neighbors in standardized feature space."""

    labeled = _labeled_mask(y_obs)
    if not np.any(labeled):
        raise ValueError("No labeled observations in y_obs.")
    n_lab = int(np.sum(labeled))
    k = min(int(knn_k), n_lab)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs_lab = Xs[labeled]
    y_lab = y_obs[labeled].astype(float)

    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(Xs_lab)

    y_full = y_obs.copy().astype(float)
    missing = ~labeled
    if np.any(missing):
        dist, idx = nn.kneighbors(Xs[missing], n_neighbors=k, return_distance=True)
        # idx is (n_missing, k) into labeled set
        neigh_y = y_lab[idx]
        y_full[missing] = np.mean(neigh_y, axis=1)
    y_bin = y_full.copy()
    y_bin[missing] = (y_full[missing] >= 0.5).astype(int)
    y_bin[labeled] = y_obs[labeled]
    return X, y_bin.astype(int)


def _pad_to_n_features(X: np.ndarray, n_cols: int) -> np.ndarray:
    """Right-pad with zeros so X has ``n_cols`` columns (e.g. S=0 for val/test)."""

    if X.shape[1] == n_cols:
        return X
    if X.shape[1] > n_cols:
        raise ValueError(f"X has {X.shape[1]} columns but design matrix expects {n_cols}.")
    pad = n_cols - X.shape[1]
    return np.hstack([X, np.zeros((X.shape[0], pad), dtype=float)])


def _complete_y_mean_plus_S(X: np.ndarray, y_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean imputation for missing y; append column S=1 if label was missing else 0."""

    labeled = _labeled_mask(y_obs)
    if not np.any(labeled):
        raise ValueError("No labeled observations in y_obs.")
    p = float(np.mean(y_obs[labeled]))
    fill = 1 if p >= 0.5 else 0
    y_full = y_obs.copy()
    y_full[~labeled] = fill
    S = (~labeled).astype(float).reshape(-1, 1)
    X_aug = np.hstack([X, S])
    return X_aug, y_full.astype(int)


class UnlabeledLogReg:
    """Logistic Lasso (FISTA) after completing missing binary labels in y_obs (-1 = missing).

    Training uses ``LogisticLassoFISTA`` on fully observed (X, y_hat). Validation data
    must have real labels in {{0,1}} for lambda selection.

    For ``mean_plus_S``, an extra feature column S indicates whether the training label
    was originally missing. Validation and test matrices are zero-padded on the right
    to match the training design (S = 0 when the label is observed).
    """

    def __init__(
        self,
        completion_method: CompletionMethod = "mean",
        *,
        knn_k: int = 5,
        **fista_kwargs,
    ) -> None:
        self.completion_method: CompletionMethod = completion_method
        self.knn_k = int(knn_k)
        self.fista_kwargs = fista_kwargs
        self.model_: Optional[LogisticLassoFISTA] = None

    def _build_completed(self, X: np.ndarray, y_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.completion_method == "mean":
            return _complete_y_mean(X, y_obs)
        if self.completion_method == "knn_mean":
            return _complete_y_knn_mean(X, y_obs, knn_k=self.knn_k)
        if self.completion_method == "mean_plus_S":
            return _complete_y_mean_plus_S(X, y_obs)
        raise ValueError(f"Unknown completion_method: {self.completion_method}")

    def fit(
        self,
        X_train: ArrayLike,
        y_obs: ArrayLike,
        *,
        X_valid: ArrayLike,
        y_valid: ArrayLike,
        metric: str = "balanced_accuracy",
        threshold: float = 0.5,
        **fit_kwargs,
    ) -> "UnlabeledLogReg":
        Xtr = _as_2d_float_array(X_train)
        yv = _as_1d_int_array(y_obs)
        if np.any((yv != -1) & ((yv < 0) | (yv > 1))):
            raise ValueError("y_obs must be in {0,1} or -1 for missing.")

        Xc, yc = self._build_completed(Xtr, yv)
        n_design = int(Xc.shape[1])
        Xva = _pad_to_n_features(_as_2d_float_array(X_valid), n_design)
        self.model_ = LogisticLassoFISTA(**self.fista_kwargs)
        self.model_.fit(Xc, yc, X_valid=Xva, y_valid=y_valid, metric=metric, threshold=threshold, **fit_kwargs)
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        X_arr = _as_2d_float_array(X)
        n_feat = int(self.model_.coef_.shape[0])
        X_arr = _pad_to_n_features(X_arr, n_feat)
        return self.model_.predict_proba(X_arr)
    
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict class labels for samples in X."""
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        
        X_arr = _as_2d_float_array(X)
        n_feat = int(self.model_.coef_.shape[0])
        X_arr = _pad_to_n_features(X_arr, n_feat)
        
        return self.model_.predict(X_arr)
