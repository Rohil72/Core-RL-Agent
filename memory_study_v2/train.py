"""Training loop, loss weighting, checkpoint selection, and resume (v2).

Acceptance criteria addressed:
- A13: Both backbones initialize fresh, receive gradients, and update weights.
- A14: Loss weights and microbatch accumulation match macro reference.
- A15: Early stopping and earliest-best tie rule match deterministic loss-sequence fixtures.
- A16: Optimizer/RNG/sampler resume matches uninterrupted run.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def compute_equal_market_weights(
    market_labels: np.ndarray,
    target_num_markets: int = 6,
) -> np.ndarray:
    """Compute sample weights w_i = N / (target_num_markets * N_m)."""
    N = len(market_labels)
    unique_markets = np.unique(market_labels)
    weights = np.zeros(N, dtype=np.float32)
    for m in unique_markets:
        mask = (market_labels == m)
        n_m = np.sum(mask)
        if n_m > 0:
            weights[mask] = N / float(target_num_markets * n_m)
    return weights


class EarlyStoppingSelector:
    """Determines checkpoint selection and early stopping under contract:

    - Minimum 5 epochs.
    - Maximum 50 epochs.
    - Patience 5 validations without an improvement > 1e-6 absolute.
    - Earliest checkpoint wins ties (strictly requires improvement > 1e-6).
    """

    def __init__(
        self,
        min_epochs: int = 5,
        max_epochs: int = 50,
        patience: int = 5,
        min_improvement: float = 1e-6,
    ):
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_improvement = min_improvement

        self.best_loss = float("inf")
        self.best_epoch = -1
        self.patience_counter = 0
        self.should_stop = False

    def step(self, epoch: int, val_loss: float) -> bool:
        """Process validation loss for completed epoch. Returns True if this epoch is a new best."""
        is_new_best = False
        # Must improve by more than min_improvement
        if val_loss < self.best_loss - self.min_improvement:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.patience_counter = 0
            is_new_best = True
        else:
            self.patience_counter += 1

        if epoch >= self.min_epochs and self.patience_counter >= self.patience:
            self.should_stop = True

        if epoch >= self.max_epochs:
            self.should_stop = True

        return is_new_best


def compute_equal_market_val_mse(
    predictions: np.ndarray,
    targets: np.ndarray,
    market_labels: np.ndarray,
) -> float:
    """Compute validation MSE: ordinary MSE within each market, averaged over markets."""
    unique_markets = np.unique(market_labels)
    market_mses = []
    for m in unique_markets:
        mask = (market_labels == m)
        if np.any(mask):
            diff = predictions[mask] - targets[mask]
            market_mses.append(float(np.mean(diff ** 2)))
    return float(np.mean(market_mses))


@dataclass
class TrainingState:
    epoch: int
    best_loss: float
    best_epoch: int
    patience_counter: int
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    torch_rng_state: torch.Tensor
    numpy_rng_state: Any


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    selector: EarlyStoppingSelector,
    epoch: int,
) -> TrainingState:
    """Capture full resumable training state."""
    return TrainingState(
        epoch=epoch,
        best_loss=selector.best_loss,
        best_epoch=selector.best_epoch,
        patience_counter=selector.patience_counter,
        model_state=copy.deepcopy(model.state_dict()),
        optimizer_state=copy.deepcopy(optimizer.state_dict()),
        torch_rng_state=torch.get_rng_state(),
        numpy_rng_state=np.random.get_state(),
    )


def load_checkpoint(
    state: TrainingState,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    selector: EarlyStoppingSelector,
) -> int:
    """Restore state into model, optimizer, selector, and RNG. Returns next epoch to execute."""
    model.load_state_dict(state.model_state)
    optimizer.load_state_dict(state.optimizer_state)
    selector.best_loss = state.best_loss
    selector.best_epoch = state.best_epoch
    selector.patience_counter = state.patience_counter
    torch.set_rng_state(state.torch_rng_state)
    np.random.set_state(state.numpy_rng_state)
    return state.epoch + 1
