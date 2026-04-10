from __future__ import annotations

import numpy as np
from numpy.random import Generator
from scipy.optimize import bisect
import pandas as pd


def mcar(df, c=0.2, rng: Generator | None = None):
    """
    Generates missingness in the 'class' column using a Missing Completely At Random (MCAR) mechanism.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset. Must contain a column named 'class'.
    c : float, default 0.2
        The target fraction of missing values to generate (0.0 to 1.0).
    rng : np.random.Generator, optional
        If given, used for Bernoulli draws; otherwise ``numpy.random.rand``.

    Returns
    -------
    df_out : pd.DataFrame
        A copy of df where 'class' is replaced by 'class_observed'.
        Missing values are encoded as -1.
    actual_fraction : float
        The actual proportion of missingness generated (will be close to 'c').
    missing_mask : ndarray of bool, shape (n_rows,)
        True where the label was made missing.
    """
    df_out = df.copy()
    n = df_out.shape[0]
    if rng is not None:
        missing_mask = rng.random(n) < c
    else:
        missing_mask = np.random.rand(n) < c

    df_out["class_observed"] = df_out["class"]
    df_out.loc[missing_mask, "class_observed"] = -1
    df_out.drop(columns=["class"], inplace=True)
    return df_out, missing_mask.mean(), missing_mask

def generate_missing(
    target_fraction,
    df,
    betas_x=None,
    column_index=None,
    beta_strength=0.8,
    mode="MAR",
    y_strength_scale=1,
    rng: Generator | None = None,
):
    """
    Generates a missingness mask for the 'class' column using a logistic regression model.
    
    The probability of a value being missing is calculated as:
    P(missing) = 1 / (1 + exp(-(beta0 + Σ(beta_i * x_i))))

    IMPORTANT: Input features in 'df' (excluding 'class') should be standardized 
    (mean=0, std=1) before calling this function to ensure stable and predictable 
    missingness patterns regardless of feature scales.

    Parameters:
    -----------
    target_fraction : float
        The desired proportion of missing data (e.g., 0.2 for 20%).
    df : pd.DataFrame
        The dataset. Must contain a column named 'class'.
    betas_x : np.array, optional
        Weights for the features. If None, defaults to beta_strength for all features.
    column_index : int, optional
        Used for MAR1 scenarios to isolate missingness dependency to a single feature index.
    beta_strength : float, default 0.8
        The default magnitude for feature coefficients.
    mode : str, default 'MAR'
        'MAR': Missingness depends only on observed features (X).
        'MNAR': Missingness depends on both observed features (X) and the target variable (Y).
    y_strength_scale : float, default 1
        Multiplier for the 'class' coefficient in MNAR mode to increase/decrease
        the target variable's influence on its own missingness.
    rng : np.random.Generator, optional
        If given, used for Bernoulli draws; otherwise ``numpy.random.rand``.

    Mechanism Descriptions:
    -----------------------
    - MAR1 (Single Variable): Missingness in 'class' is driven by one specific observed 
      feature (specified via column_index).
    - MAR2 (Multi-Variable): Missingness in 'class' is driven by multiple or all 
      observed features (Uniform if betas are equal, Random if betas vary).
    - MNAR (Missing Not At Random): Missingness in 'class' is driven by the value of 
      'class' in combination with other observed features.

    Returns:
    --------
    df_out : pd.DataFrame
        A copy of df where 'class' is replaced by 'class_observed' (missing values marked as -1).
    beta0 : float
        The intercept calculated to achieve the target_fraction.
    final_betas : np.array
        The coefficients used in the logistic model.
    actual_fraction : float
        The actual proportion of missingness achieved in this run.
    missing_mask : ndarray of bool, shape (n_rows,)
        True where the label was made missing.
    """
    if betas_x is None:
        n_features = df.shape[1] - 1
        betas_x = np.ones(n_features) * beta_strength
    
    class_idx = df.columns.get_loc('class')

    if mode == 'MNAR':
        df_selected = df
        max_beta = max(betas_x) if len(betas_x) > 0 else beta_strength
        betas = np.insert(betas_x, class_idx, max_beta * y_strength_scale)
    else:
        df_selected = df.drop(columns=['class'])
        betas = betas_x

    beta0, final_betas = beta_adjust(target_fraction, df_selected, betas, column_index, beta_strength)
    linear_comp = np.dot(df_selected, final_betas)
    probs = 1 / (1 + np.exp(-(beta0 + linear_comp)))
    n_row = df_selected.shape[0]
    if rng is not None:
        missing_mask = rng.random(n_row) < probs
    else:
        missing_mask = np.random.rand(n_row) < probs

    df_out = df.copy()
    df_out["class_observed"] = df_out["class"]
    df_out.loc[missing_mask, "class_observed"] = -1
    df_out.drop(columns=["class"], inplace=True)

    return df_out, beta0, final_betas, missing_mask.mean(), missing_mask

def beta_adjust(target_fraction, X, betas=None, column_index=None, beta_strength=0.8):
    """
    Calibrates the logistic intercept (beta0) to achieve a specific missingness proportion.

    This function solves for the intercept in a logistic model such that the expected 
    value of the sigmoid function over the dataset matches the target_fraction. It 
    uses the Bisection Method to find the root of the objective function.

    Parameters:
    -----------
    target_fraction : float
        The desired mean probability of missingness (e.g., 0.2).
    X : pd.DataFrame or np.array
        The feature matrix (should be standardized).
    betas : np.array, optional
        Custom weights for features. If None, beta_strength is used.
    column_index : int or str, optional
        The specific feature index to use for MAR1 scenarios. If set, all other 
        weights are zeroed out.
    beta_strength : float, default 0.8
        The magnitude applied to the active features.

    Returns:
    --------
    optimal_beta0 : float
        The intercept value that aligns the model with the target_fraction.
    weights : np.array
        The final coefficient vector used for the linear combination.
    """
    if isinstance(column_index, str):
        column_index = X.columns.get_loc(column_index)
    if column_index is not None:
        weights = np.zeros(X.shape[1])
        weights[column_index] = beta_strength
    else:
        weights = betas if betas is not None else np.ones(X.shape[1]) * beta_strength

    linear_comp = np.dot(X, weights)

    def objective(beta0):
        z = np.clip(beta0 + linear_comp, -500, 500)
        probs = 1 / (1 + np.exp(-z))
        return np.mean(probs) - target_fraction

    a, b = -5000, 5000
    if objective(a) * objective(b) > 0:
        a, b = -1e6, 1e6
        
    optimal_beta0 = bisect(objective, a, b)
    return optimal_beta0, weights