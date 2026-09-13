"""Ridge regression with equal-market weighting in float64 (v2).

Solves the weighted centered normal equations with unpenalized intercept:
objective = equal_market_MSE + 0.001 * ||beta||_2^2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RidgeModel:
    coef: np.ndarray      # shape (966,), float64
    intercept: float      # float64
    l2_lambda: float = 0.001

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous returns. Input shape (B, 966). Returns (B,)."""
        X64 = X.astype(np.float64)
        return np.dot(X64, self.coef) + self.intercept


def fit_ridge_model(
    X: np.ndarray,
    y: np.ndarray,
    market_labels: np.ndarray,
    l2_lambda: float = 0.001,
) -> RidgeModel:
    """Fit deterministic Ridge regression in float64 with equal-market sample weighting.

    Objective:
        (1/N) * sum_i w_i * (y_i - (beta_0 + x_i' * beta))^2 + l2_lambda * ||beta||_2^2
    where w_i = N / (6 * N_m).
    """
    X64 = X.astype(np.float64)
    y64 = y.astype(np.float64)
    N, D = X64.shape

    # Calculate equal-market sample weights
    unique_markets = np.unique(market_labels)
    num_markets = len(unique_markets)
    weights = np.zeros(N, dtype=np.float64)
    for m in unique_markets:
        m_mask = (market_labels == m)
        n_m = np.sum(m_mask)
        if n_m > 0:
            weights[m_mask] = N / (float(num_markets) * float(n_m))

    sum_w = np.sum(weights)

    # Weighted means
    x_mean = np.sum(X64 * weights[:, None], axis=0) / sum_w  # (D,)
    y_mean = np.sum(y64 * weights) / sum_w                   # scalar

    # Centered data
    Xc = X64 - x_mean  # (N, D)
    yc = y64 - y_mean  # (N,)

    # Weighted scatter: (1/N) * Xc' W Xc
    # (Xc * weights[:, None])' @ Xc / N
    X_weighted = Xc * weights[:, None]
    A = (X_weighted.T @ Xc) / float(N) + l2_lambda * np.eye(D, dtype=np.float64)
    b = (X_weighted.T @ yc) / float(N)

    coef = np.linalg.solve(A, b)
    intercept = float(y_mean - np.dot(x_mean, coef))

    return RidgeModel(coef=coef, intercept=intercept, l2_lambda=l2_lambda)
