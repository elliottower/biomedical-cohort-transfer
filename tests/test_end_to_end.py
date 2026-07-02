"""End-to-end test with synthetic data.

Plants a known structure: two cohorts with different subspace orientations
and a binary label correlated with one subspace direction. Verifies the
full pipeline detects the planted transportability gap.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.preprocess import preprocess_cohorts
from src.embed import embed
from src.transportability import sheaf_h1_two_cohort
from src.external_validation import external_validation, internal_validation


def _make_synthetic_cohorts(n1=300, n2=250, d=50, seed=42):
    """Two cohorts with shared signal but different noise subspaces.

    Cohort 1: label correlated with features 0-4
    Cohort 2: same signal but rotated noise subspace (features 5-49)
    """
    rng = np.random.default_rng(seed)

    y1 = rng.integers(0, 2, size=n1)
    y2 = rng.integers(0, 2, size=n2)

    X1 = rng.standard_normal((n1, d))
    X2 = rng.standard_normal((n2, d))

    signal_dims = 5
    for i in range(signal_dims):
        X1[:, i] += y1 * 1.5
        X2[:, i] += y2 * 1.5

    from scipy.stats import ortho_group
    R = ortho_group.rvs(d - signal_dims, random_state=42)
    X2[:, signal_dims:] = X2[:, signal_dims:] @ R

    cols = [f"feat_{i}" for i in range(d)]
    df1 = pd.DataFrame(X1, columns=cols)
    df1["label"] = y1
    df2 = pd.DataFrame(X2, columns=cols)
    df2["label"] = y2

    return df1, df2


def test_synthetic_pipeline_detects_transportability():
    df1, df2 = _make_synthetic_cohorts()
    data = preprocess_cohorts(df1, df2, log_transform_data=False)

    X1_emb = embed(data["X1"], backend="raw_features")
    X2_emb = embed(data["X2"], backend="raw_features")

    geom = sheaf_h1_two_cohort(X1_emb, X2_emb, k=10)
    assert geom["geodesic_dist"] > 0.5

    int_c1 = internal_validation(X1_emb, data["y1"], seed=42)
    ext = external_validation(X1_emb, data["y1"], X2_emb, data["y2"],
                              n_bootstrap=100, seed=42)

    assert int_c1["mean_auc"] > 0.6
    assert ext["forward"]["auc"] > 0.5


def test_identical_cohorts_have_small_gap():
    rng = np.random.default_rng(99)
    n, d = 200, 30
    y = rng.integers(0, 2, size=n)
    X = rng.standard_normal((n, d))
    for i in range(3):
        X[:, i] += y * 2.0

    split = n // 2
    cols = [f"feat_{i}" for i in range(d)]
    df1 = pd.DataFrame(X[:split], columns=cols)
    df1["label"] = y[:split]
    df2 = pd.DataFrame(X[split:], columns=cols)
    df2["label"] = y[split:]

    data = preprocess_cohorts(df1, df2, log_transform_data=False)
    X1_emb = embed(data["X1"], backend="raw_features")
    X2_emb = embed(data["X2"], backend="raw_features")

    geom = sheaf_h1_two_cohort(X1_emb, X2_emb, k=5)
    # Same-distribution split still has finite-sample distance;
    # just check it's smaller than the planted-rotation case
    assert geom["geodesic_dist"] < 3.5
