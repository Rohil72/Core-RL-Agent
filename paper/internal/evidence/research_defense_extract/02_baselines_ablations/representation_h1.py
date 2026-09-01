"""H1 Representation Diagnostics Suite.

Implements rigorous representation evaluation comparing learned 128-dimensional
embeddings against standardized raw features and dimension-matched PCA controls:
1. Linear Centered Kernel Alignment (CKA) across random seeds.
2. Seed-to-seed k-nearest-neighbour overlap (Jaccard similarity).
3. Retrieval outcome homogeneity across neighbours (MAE/MSE of matured returns).
4. Nuisance identity confounding: cross-validated prediction of ticker & market.
5. Incremental outcome association controlling for ticker, market, date block,
   volatility, and momentum.
6. Non-neural PCA control fitted solely on training split.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("representation_h1")


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Linear Centered Kernel Alignment (CKA) between two feature matrices.
    
    Both X and Y should have shape [N, D1] and [N, D2].
    Invariant to orthogonal transformations and isotropic scaling.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Sample counts must match: {X.shape[0]} vs {Y.shape[0]}")
    if X.shape[0] < 2:
        return 1.0

    # Center columns
    X_c = X - np.mean(X, axis=0, keepdims=True)
    Y_c = Y - np.mean(Y, axis=0, keepdims=True)

    # Compute HSIC with linear kernel
    # HSIC(K, L) = (1 / (n - 1)^2) * tr(K H L H) = (1 / (n - 1)^2) * ||Y_c^T X_c||_F^2
    dot_prod = Y_c.T @ X_c
    hsic = np.sum(dot_prod**2)

    # Normalization terms
    norm_x = np.sum((X_c.T @ X_c) ** 2)
    norm_y = np.sum((Y_c.T @ Y_c) ** 2)

    denom = np.sqrt(norm_x * norm_y)
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(hsic / denom, 0.0, 1.0))


def compute_knn_overlap(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    k: int = 25,
) -> float:
    """Compute mean Jaccard overlap of k-nearest neighbours across two embeddings."""
    n = len(embeddings_a)
    if n <= 1:
        return 1.0
    actual_k = min(k, n - 1)

    nn_a = NearestNeighbors(n_neighbors=actual_k + 1, metric="euclidean").fit(embeddings_a)
    nn_b = NearestNeighbors(n_neighbors=actual_k + 1, metric="euclidean").fit(embeddings_b)

    indices_a = nn_a.kneighbors(return_distance=False)[:, 1:]  # Exclude self
    indices_b = nn_b.kneighbors(return_distance=False)[:, 1:]

    jaccards = []
    for i in range(n):
        set_a = set(indices_a[i])
        set_b = set(indices_b[i])
        union = len(set_a | set_b)
        if union == 0:
            continue
        jaccards.append(len(set_a & set_b) / union)

    return float(np.mean(jaccards)) if jaccards else 0.0


def compute_neighbour_outcome_homogeneity(
    embeddings: np.ndarray,
    outcome_series: np.ndarray,
    k: int = 25,
) -> dict[str, float]:
    """Evaluate how homogeneous outcomes are within the k-nearest neighbours.
    
    Lower outcome variance / MAE indicates higher semantic coherence.
    """
    n = len(embeddings)
    if n <= 1:
        return {"neighbour_outcome_mae": 0.0, "neighbour_outcome_var": 0.0}
    actual_k = min(k, n - 1)

    nn = NearestNeighbors(n_neighbors=actual_k + 1, metric="euclidean").fit(embeddings)
    indices = nn.kneighbors(return_distance=False)[:, 1:]

    outcomes = np.asarray(outcome_series, dtype=float)
    valid_mask = np.isfinite(outcomes)

    maes = []
    vars_ = []
    for i in range(n):
        if not valid_mask[i]:
            continue
        nbr_idx = indices[i]
        nbr_outcomes = outcomes[nbr_idx]
        nbr_valid = nbr_outcomes[np.isfinite(nbr_outcomes)]
        if len(nbr_valid) == 0:
            continue
        maes.append(float(np.mean(np.abs(nbr_valid - outcomes[i]))))
        vars_.append(float(np.var(nbr_valid)))

    return {
        "neighbour_outcome_mae": float(np.mean(maes)) if maes else 0.0,
        "neighbour_outcome_var": float(np.mean(vars_)) if vars_ else 0.0,
    }


def evaluate_nuisance_identity_predictability(
    embeddings: np.ndarray,
    ticker_labels: Sequence[str],
    market_labels: Sequence[str] | None = None,
    cv_folds: int = 5,
    random_seed: int = 7,
) -> dict[str, float]:
    """Measure how easily ticker identity and market identity can be decoded from embeddings.
    
    High accuracy indicates that the representation is largely memorizing identity rather
    than learning invariant market dynamics.
    """
    tickers = np.asarray(ticker_labels)
    unique_tickers, counts = np.unique(tickers, return_counts=True)
    # Only evaluate if multiple tickers and sufficient samples per class
    min_count = np.min(counts) if len(counts) > 0 else 0
    actual_folds = min(cv_folds, min_count) if min_count >= 2 else 2

    ticker_acc = 0.0
    if len(unique_tickers) > 1 and min_count >= 2:
        clf = LogisticRegression(C=1.0, max_iter=200, random_state=random_seed)
        scores = cross_val_score(clf, embeddings, tickers, cv=actual_folds, scoring="accuracy")
        ticker_acc = float(np.mean(scores))

    market_acc = 0.0
    if market_labels is not None:
        markets = np.asarray(market_labels)
        unique_markets, m_counts = np.unique(markets, return_counts=True)
        m_min = np.min(m_counts) if len(m_counts) > 0 else 0
        m_folds = min(cv_folds, m_min) if m_min >= 2 else 2
        if len(unique_markets) > 1 and m_min >= 2:
            clf_m = LogisticRegression(C=1.0, max_iter=200, random_state=random_seed)
            m_scores = cross_val_score(clf_m, embeddings, markets, cv=m_folds, scoring="accuracy")
            market_acc = float(np.mean(m_scores))

    return {
        "ticker_decodability_accuracy": ticker_acc,
        "market_decodability_accuracy": market_acc,
    }


def fit_pca_control(
    train_features: np.ndarray,
    n_components: int = 128,
    random_seed: int = 7,
) -> PCA:
    """Fit a dimension-matched PCA control projection strictly on training features."""
    max_comp = min(train_features.shape[0], train_features.shape[1], n_components)
    pca = PCA(n_components=max_comp, random_state=random_seed)
    pca.fit(train_features)
    return pca


def evaluate_incremental_outcome_association(
    neighbour_predictions: np.ndarray,
    actual_outcomes: np.ndarray,
    confounder_matrix: np.ndarray,
) -> dict[str, float]:
    """Test whether neighbour predictions add explanatory power beyond confounders.
    
    Controls for: ticker fixed effects, market indicators, volatility, momentum.
    Compares R2 of base model (confounders only) vs full model (confounders + memory).
    """
    valid = np.isfinite(neighbour_predictions) & np.isfinite(actual_outcomes)
    if confounder_matrix.ndim == 2:
        valid &= np.isfinite(confounder_matrix).all(axis=1)

    y = actual_outcomes[valid]
    pred = neighbour_predictions[valid]
    conf = confounder_matrix[valid]

    if len(y) < 20:
        return {"base_r2": 0.0, "full_r2": 0.0, "incremental_r2": 0.0, "p_value": 1.0}

    # Model 1: Confounders only
    ridge_base = Ridge(alpha=1.0)
    ridge_base.fit(conf, y)
    y_pred_base = ridge_base.predict(conf)
    base_r2 = float(r2_score(y, y_pred_base))

    # Model 2: Confounders + Memory Prediction
    X_full = np.column_stack([conf, pred])
    ridge_full = Ridge(alpha=1.0)
    ridge_full.fit(X_full, y)
    y_pred_full = ridge_full.predict(X_full)
    full_r2 = float(r2_score(y, y_pred_full))

    incremental_r2 = max(0.0, full_r2 - base_r2)

    return {
        "base_r2": base_r2,
        "full_r2": full_r2,
        "incremental_r2": incremental_r2,
        "p_value": None,  # In-sample Ridge does not yield a classical p-value; use bootstrap
    }
