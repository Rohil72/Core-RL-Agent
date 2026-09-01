"""Tests for H1 representation evaluation diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA

from src.eval.representation_h1 import (
    compute_knn_overlap,
    compute_neighbour_outcome_homogeneity,
    evaluate_incremental_outcome_association,
    evaluate_nuisance_identity_predictability,
    fit_pca_control,
    linear_cka,
)


def test_linear_cka_properties():
    np.random.seed(42)
    X = np.random.randn(100, 32)
    # Self-CKA must be exactly 1.0
    cka_self = linear_cka(X, X)
    assert np.isclose(cka_self, 1.0, atol=1e-5)

    # Invariant to orthogonal rotation: Y = X @ R
    Q, _ = np.linalg.qr(np.random.randn(32, 32))
    Y = X @ Q
    cka_rot = linear_cka(X, Y)
    assert np.isclose(cka_rot, 1.0, atol=1e-4)

    # Independent matrices should have low CKA
    Z = np.random.randn(100, 32)
    cka_indep = linear_cka(X, Z)
    assert cka_indep < 0.40


def test_knn_overlap():
    np.random.seed(42)
    X = np.random.randn(50, 16)
    # Overlap with itself should be 1.0
    overlap_self = compute_knn_overlap(X, X, k=5)
    assert np.isclose(overlap_self, 1.0)

    # Overlap with random noise should be low
    Y = np.random.randn(50, 16)
    overlap_rand = compute_knn_overlap(X, Y, k=5)
    assert overlap_rand < 0.30


def test_outcome_homogeneity():
    np.random.seed(42)
    X = np.zeros((40, 4))
    # Cluster 1
    X[:20] += 5.0
    # Cluster 2
    X[20:] -= 5.0
    outcomes = np.array([0.1] * 20 + [-0.1] * 20)

    homo = compute_neighbour_outcome_homogeneity(X, outcomes, k=5)
    assert homo["neighbour_outcome_var"] < 1e-4
    assert homo["neighbour_outcome_mae"] < 1e-4


def test_nuisance_predictability():
    np.random.seed(42)
    # Embeddings that strongly encode ticker
    X = np.random.randn(60, 8)
    tickers = ["AAPL"] * 20 + ["MSFT"] * 20 + ["GOOGL"] * 20
    X[:20, 0] += 10.0
    X[20:40, 1] += 10.0
    X[40:, 2] += 10.0

    res = evaluate_nuisance_identity_predictability(X, tickers, cv_folds=3)
    assert res["ticker_decodability_accuracy"] > 0.85


def test_incremental_outcome_association():
    np.random.seed(42)
    n = 100
    confounder = np.random.randn(n, 2)
    # Outcome influenced by memory prediction beyond confounders
    pred = np.random.randn(n)
    y = 0.3 * confounder[:, 0] + 0.5 * pred + np.random.randn(n) * 0.1

    res = evaluate_incremental_outcome_association(pred, y, confounder)
    assert res["full_r2"] > res["base_r2"]
    assert res["incremental_r2"] > 0.10


def test_fit_pca_control():
    np.random.seed(42)
    train_features = np.random.randn(100, 28)
    pca = fit_pca_control(train_features, n_components=16)
    assert isinstance(pca, PCA)
    projected = pca.transform(train_features)
    assert projected.shape == (100, 16)
