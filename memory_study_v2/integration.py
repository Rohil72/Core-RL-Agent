"""Selective Trust Gate, fixed mixture selectors, and alias resolution (v2).

Acceptance criteria addressed:
- A20: Three inputs only; saved development normalizer; no outcomes supplied as gate inputs.
- A21: MSE and Sharpe selectors use specified averaging, dates, ties, and all grid rows.
- A22: Lambda endpoints alias identical predictions/paths without inflating independence counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from memory_study_v2.train import compute_equal_market_weights


class GateInputError(Exception):
    """Raised when gate receives invalid dimensions, outcomes, or unauthorized channels."""
    pass


@dataclass
class GateNormalizer:
    means: np.ndarray  # shape (3,), float64
    stds: np.ndarray   # shape (3,), float64
    scale_floor: float = 1e-4

    def transform(self, z: np.ndarray) -> np.ndarray:
        std_floored = np.maximum(self.stds, self.scale_floor)
        return (z - self.means) / std_floored


class TrustGateModule(nn.Module):
    """Logistic sigmoid gate: g = sigmoid(a' z_norm + b). Prediction: (1 - g) * base + g * memory."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        # Initial bias log(1/3)
        self.b = nn.Parameter(torch.tensor(math.log(1.0 / 3.0), dtype=torch.float32))

    def forward(self, z_norm: torch.Tensor, base: torch.Tensor, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z_norm shape: (N, 3)
        logits = torch.sum(z_norm * self.a, dim=-1) + self.b
        g = torch.sigmoid(logits)  # (N,)
        pred = (1.0 - g) * base + g * memory
        return pred, g


@dataclass
class FittedTrustGate:
    normalizer: GateNormalizer
    weights_a: np.ndarray  # shape (3,)
    bias_b: float
    loss_history: List[float]

    def predict(self, base: np.ndarray, memory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate gated predictions: (1 - g) * base + g * memory and gate values g."""
        # Strictly construct 3 inputs: (|base|, |base - memory|, |memory|)
        z = np.column_stack([
            np.abs(base),
            np.abs(base - memory),
            np.abs(memory),
        ]).astype(np.float64)

        z_norm = self.normalizer.transform(z).astype(np.float32)
        logits = np.dot(z_norm, self.weights_a) + self.bias_b
        # Logistic sigmoid
        g = 1.0 / (1.0 + np.exp(-logits))
        pred = (1.0 - g) * base + g * memory
        return pred, g


def fit_trust_gate(
    dev_base_preds: np.ndarray,
    dev_memory_preds: np.ndarray,
    dev_targets: np.ndarray,
    dev_markets: np.ndarray,
    steps: int = 50,
    lr: float = 0.05,
    weight_decay: float = 0.001,
) -> FittedTrustGate:
    """Fit Selective Trust Gate on development sample using 50 Adam steps (A20).

    Inputs to gate: strictly (|base|, |base - memory|, |memory|).
    """
    N = len(dev_base_preds)
    if dev_memory_preds.shape != (N,) or dev_targets.shape != (N,):
        raise GateInputError("Mismatched dimensions in development predictions/targets")

    # Strictly 3 inputs only
    z_raw = np.column_stack([
        np.abs(dev_base_preds),
        np.abs(dev_base_preds - dev_memory_preds),
        np.abs(dev_memory_preds),
    ]).astype(np.float64)

    # Compute development population moments
    means = np.mean(z_raw, axis=0)
    stds = np.std(z_raw, axis=0, ddof=0)
    normalizer = GateNormalizer(means=means, stds=stds)

    z_norm = normalizer.transform(z_raw).astype(np.float32)

    # Convert to tensors
    z_tensor = torch.from_numpy(z_norm)
    base_tensor = torch.from_numpy(dev_base_preds.astype(np.float32))
    mem_tensor = torch.from_numpy(dev_memory_preds.astype(np.float32))
    target_tensor = torch.from_numpy(dev_targets.astype(np.float32))

    # Sample weights for equal-market development MSE
    weights = torch.from_numpy(compute_equal_market_weights(dev_markets))
    total_w = torch.sum(weights)

    module = TrustGateModule()
    optimizer = torch.optim.Adam(
        module.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    loss_history = []
    for step in range(steps):
        optimizer.zero_grad()
        preds, _ = module(z_tensor, base_tensor, mem_tensor)
        diff = preds - target_tensor
        loss = torch.sum(weights * (diff ** 2)) / total_w
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    return FittedTrustGate(
        normalizer=normalizer,
        weights_a=module.a.detach().cpu().numpy(),
        bias_b=float(module.b.detach().cpu().item()),
        loss_history=loss_history,
    )


@dataclass
class SelectedMixture:
    objective: str  # "MIX_MSE" or "MIX_SR"
    selected_lambda: float
    grid_scores: Dict[float, float]
    is_alias: bool
    alias_of: Optional[str]  # "BASE" or "MEM_SIM"


def select_mixture_mse(
    dev_base_by_seed: Dict[int, np.ndarray],  # seed -> predictions
    dev_mem_preds: np.ndarray,
    dev_targets: np.ndarray,
    dev_markets: np.ndarray,
    lambda_grid: List[float] = [0.0, 0.10, 0.25, 0.50, 1.00],
    tie_tolerance: float = 1e-12,
) -> SelectedMixture:
    """Select mixture lambda minimizing development MSE, averaging seeds within market, then markets (A21)."""
    grid_scores: Dict[float, float] = {}
    unique_markets = np.unique(dev_markets)
    seeds = list(dev_base_by_seed.keys())

    for lam in lambda_grid:
        market_mses = []
        for m in unique_markets:
            m_mask = (dev_markets == m)
            # Average across seeds within this market
            seed_mses = []
            for s in seeds:
                b_pred = dev_base_by_seed[s][m_mask]
                m_pred = dev_mem_preds[m_mask]
                y_true = dev_targets[m_mask]

                mix_pred = (1.0 - lam) * b_pred + lam * m_pred
                mse_s = float(np.mean((mix_pred - y_true) ** 2))
                seed_mses.append(mse_s)
            market_mses.append(float(np.mean(seed_mses)))
        # Average across markets
        grid_scores[lam] = float(np.mean(market_mses))

    # Find minimum score with tie-breaking (smallest lambda wins ties within tie_tolerance)
    best_lam = lambda_grid[0]
    best_score = grid_scores[best_lam]

    for lam in lambda_grid[1:]:
        score = grid_scores[lam]
        # Must improve by more than tie_tolerance to beat an earlier (smaller) lambda
        if score < best_score - tie_tolerance:
            best_score = score
            best_lam = lam

    # Alias check (A22)
    is_alias = False
    alias_of = None
    if best_lam == 0.0:
        is_alias = True
        alias_of = "BASE"
    elif best_lam == 1.0:
        is_alias = True
        alias_of = "MEM_SIM"

    return SelectedMixture(
        objective="MIX_MSE",
        selected_lambda=best_lam,
        grid_scores=grid_scores,
        is_alias=is_alias,
        alias_of=alias_of,
    )


def select_mixture_sr(
    dev_base_by_seed: Dict[int, np.ndarray],
    dev_mem_preds: np.ndarray,
    dev_targets: np.ndarray,
    dev_markets: np.ndarray,
    lambda_grid: List[float] = [0.0, 0.10, 0.25, 0.50, 1.00],
    tie_tolerance: float = 1e-12,
    roundtrip_cost: float = 0.003,
    dev_eval_fn: Optional[Callable[[float, int, str], float]] = None,
) -> SelectedMixture:
    """Select mixture lambda maximizing development net portfolio Sharpe ratio (A21 / R07).

    Averages net Sharpe across seeds within each market, then across markets.
    Tie-breaking: smallest lambda wins ties within tie_tolerance.
    """
    grid_scores: Dict[float, float] = {}
    unique_markets = np.unique(dev_markets)
    seeds = list(dev_base_by_seed.keys())

    for lam in lambda_grid:
        market_sharpes = []
        for m in unique_markets:
            m_mask = (dev_markets == m)
            seed_sharpes = []
            for s in seeds:
                if dev_eval_fn is not None:
                    sr_val = dev_eval_fn(lam, s, str(m))
                else:
                    b_pred = dev_base_by_seed[s][m_mask]
                    m_pred = dev_mem_preds[m_mask]
                    y_true = dev_targets[m_mask]

                    mix_pred = (1.0 - lam) * b_pred + lam * m_pred
                    # Active positive forecast rule: entry when mix_pred > 0
                    active = (mix_pred > 0.0)
                    if np.any(active):
                        # Realized net return deducting roundtrip fee and slippage (2 * 0.0015 = 0.003)
                        net_ret = np.where(active, y_true - roundtrip_cost, 0.0)
                    else:
                        net_ret = np.zeros_like(y_true)

                    r_mean = float(np.mean(net_ret))
                    r_std = float(np.std(net_ret, ddof=0))
                    if r_std > 1e-8:
                        sr_val = (r_mean / r_std) * math.sqrt(252.0)
                    else:
                        sr_val = 0.0

                seed_sharpes.append(sr_val)
            market_sharpes.append(float(np.mean(seed_sharpes)))
        grid_scores[lam] = float(np.mean(market_sharpes))

    # Maximize Sharpe with tie-breaking (smallest lambda wins ties within tie_tolerance)
    best_lam = lambda_grid[0]
    best_score = grid_scores[best_lam]

    for lam in lambda_grid[1:]:
        score = grid_scores[lam]
        # Must strictly beat earlier smaller lambda by more than tie_tolerance
        if score > best_score + tie_tolerance:
            best_score = score
            best_lam = lam

    # Alias check (A22)
    is_alias = False
    alias_of = None
    if best_lam == 0.0:
        is_alias = True
        alias_of = "BASE"
    elif best_lam == 1.0:
        is_alias = True
        alias_of = "MEM_SIM"

    return SelectedMixture(
        objective="MIX_SR",
        selected_lambda=best_lam,
        grid_scores=grid_scores,
        is_alias=is_alias,
        alias_of=alias_of,
    )

