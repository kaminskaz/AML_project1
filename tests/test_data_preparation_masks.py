"""Smoke tests for missingness masks returned by data_preparation."""

import numpy as np
import pandas as pd
from src.data_preparation import generate_missing, mcar


def test_mcar_returns_mask_of_correct_length():
    df = pd.DataFrame({"f0": [0.0, 1.0, -1.0], "f1": [0.0, 0.5, 1.0], "class": [0, 1, 0]})
    rng = np.random.default_rng(42)
    out, frac, mask = mcar(df, c=0.3, rng=rng)
    assert len(mask) == 3
    assert mask.dtype == bool
    assert abs(float(mask.mean()) - float(frac)) < 1e-15
    assert "class_observed" in out.columns


def test_generate_missing_returns_mask():
    df = pd.DataFrame({"f0": np.linspace(-1, 1, 50), "class": np.random.randint(0, 2, 50)})
    rng = np.random.default_rng(0)
    out, b0, fb, frac, mask = generate_missing(
        0.2, df, None, 0, 0.8, "MAR", 1, rng=rng
    )
    assert len(mask) == 50
    assert abs(mask.mean() - frac) < 1e-15
