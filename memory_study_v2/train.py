"""Training loop, loss weighting, checkpoint selection, runner, and resume (v2).

Acceptance criteria addressed:
- A13: Both backbones initialize fresh, receive gradients, and update weights.
- A14: Loss weights and microbatch accumulation match macro reference.
- A15: Early stopping and earliest-best tie rule match deterministic loss-sequence fixtures.
- A16: Full optimizer/RNG/sampler resume matches uninterrupted run (R06).
- Complete integrated training runner for all 36 fits with per-fold sample counts (R02).
"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    return float(np.mean(market_mses)) if market_mses else 0.0


@dataclass
class TrainingState:
    """Complete resumable training state (R06)."""
    epoch: int
    macro_step: int
    micro_step: int
    best_loss: float
    best_epoch: int
    patience_counter: int
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    torch_cpu_rng_state: torch.Tensor
    torch_cuda_rng_state: Optional[List[torch.Tensor]]
    numpy_rng_state: Any
    python_rng_state: Any
    generator_state: Optional[Dict[str, Any]] = None
    epoch_val_losses: List[float] = field(default_factory=list)
    sampler_cursor: int = 0
    epoch_permutation: Optional[List[int]] = None


def capture_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    selector: EarlyStoppingSelector,
    epoch: int,
    macro_step: int = 0,
    micro_step: int = 0,
    sampler_cursor: int = 0,
    generator: Optional[np.random.Generator] = None,
    epoch_val_losses: Optional[List[float]] = None,
    epoch_permutation: Optional[List[int]] = None,
) -> TrainingState:
    """Capture complete state including CUDA, Python, NumPy RNGs, Generator, and cursor (R06)."""
    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()

    gen_state = generator.bit_generator.state if generator is not None else None

    return TrainingState(
        epoch=epoch,
        macro_step=macro_step,
        micro_step=micro_step,
        best_loss=selector.best_loss,
        best_epoch=selector.best_epoch,
        patience_counter=selector.patience_counter,
        model_state=copy.deepcopy(model.state_dict()),
        optimizer_state=copy.deepcopy(optimizer.state_dict()),
        torch_cpu_rng_state=torch.get_rng_state(),
        torch_cuda_rng_state=cuda_state,
        numpy_rng_state=np.random.get_state(),
        python_rng_state=random.getstate(),
        generator_state=gen_state,
        epoch_val_losses=list(epoch_val_losses) if epoch_val_losses is not None else [],
        sampler_cursor=sampler_cursor,
        epoch_permutation=list(epoch_permutation) if epoch_permutation is not None else None,
    )


def restore_training_state(
    state: TrainingState,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    selector: EarlyStoppingSelector,
    generator: Optional[np.random.Generator] = None,
) -> int:
    """Restore complete state into model, optimizer, selector, and RNGs (R06)."""
    model.load_state_dict(state.model_state)
    optimizer.load_state_dict(state.optimizer_state)
    selector.best_loss = state.best_loss
    selector.best_epoch = state.best_epoch
    selector.patience_counter = state.patience_counter

    torch.set_rng_state(state.torch_cpu_rng_state)
    if state.torch_cuda_rng_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.torch_cuda_rng_state)
    np.random.set_state(state.numpy_rng_state)
    random.setstate(state.python_rng_state)

    if state.generator_state is not None and generator is not None:
        generator.bit_generator.state = state.generator_state

    if getattr(state, "sampler_cursor", 0) > 0:
        return state.epoch
    return state.epoch + 1


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    selector: EarlyStoppingSelector,
    epoch: int,
) -> TrainingState:
    """Legacy helper returning TrainingState."""
    return capture_training_state(model, optimizer, selector, epoch)


def load_checkpoint(
    state: TrainingState,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    selector: EarlyStoppingSelector,
) -> int:
    """Legacy helper restoring TrainingState."""
    return restore_training_state(state, model, optimizer, selector)


@dataclass
class TrainingSummary:
    model_type: str
    seed: int
    best_epoch: int
    best_loss: float
    epochs_trained: int
    total_macro_steps: int
    samples_per_market: Dict[str, int]
    epoch_val_losses: List[float]
    early_stopped: bool


def load_neural_config(
    config_path: Optional[Union[str, Path]] = None,
    execution_mode: str = "pilot",
) -> Dict[str, Any]:
    """Load and validate neural training configuration from config.proposed.json (C3).

    In permissive 'pilot' mode, missing config files use standard defaults.
    In fail-closed 'production' mode, the config file is mandatory and production_authorized must be true.
    """
    if config_path is None:
        p = Path("rebuild_plan/config.proposed.json")
    else:
        p = Path(config_path)

    if not p.exists():
        if execution_mode == "production":
            raise FileNotFoundError(f"In production mode, configuration file '{p}' is mandatory.")
        return {
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "epsilon": 1e-08,
            "effective_batch": 512,
            "max_epochs": 50,
            "min_epochs": 5,
            "early_stop_patience": 5,
            "minimum_improvement": 1e-06,
            "gradient_norm_clip": 1.0,
        }

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if execution_mode == "production":
        if not data.get("production_authorized", False):
            raise PermissionError("Production execution rejected: production_authorized is false.")

    return data.get("neural", {})


def train_backbone_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_markets: np.ndarray,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    val_markets: np.ndarray,
    seed: int = 7,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    grad_clip_norm: Optional[float] = None,
    betas: Optional[Tuple[float, float]] = None,
    eps: Optional[float] = None,
    micro_batch_size: int = 64,
    effective_batch_size: Optional[int] = None,
    min_epochs: Optional[int] = None,
    max_epochs: Optional[int] = None,
    patience: Optional[int] = None,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    interrupt_at_epoch: Optional[int] = None,
    interrupt_at_macro_step: Optional[int] = None,
    resume_from_checkpoint: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    execution_mode: str = "pilot",
) -> Tuple[nn.Module, TrainingSummary]:
    """Complete integrated dataset-to-epoch training runner (R02).

    Features:
    - Microbatch accumulation to match effective_batch_size without epoch boundary drift.
    - Equal-market sample weighting.
    - EarlyStoppingSelector with earliest-best tie rule.
    - Checkpoint persistence to disk with full RNG state and Generator state (R06).
    - Bitwise exact checkpoint resume capability.
    - AdamW with explicit weight decay (0.0001) and gradient clipping (1.0).
    - Streamed validation batches to bound peak memory.
    - Fail-closed production configuration validation (C3).
    - Macro-boundary interruption handling (C5).
    """
    # 1. Bind configuration from config.proposed.json (C3)
    n_cfg = load_neural_config(config_path, execution_mode=execution_mode)
    cfg_lr = float(n_cfg.get("learning_rate", 0.001))
    cfg_wd = float(n_cfg.get("weight_decay", 0.0001))
    cfg_clip = float(n_cfg.get("gradient_norm_clip", 1.0))
    cfg_betas = tuple(n_cfg.get("betas", [0.9, 0.999]))
    cfg_eps = float(n_cfg.get("epsilon", 1e-08))
    cfg_eb = int(n_cfg.get("effective_batch", 512))

    if execution_mode == "production":
        if lr is not None and lr != cfg_lr:
            raise ValueError(f"Unapproved hyperparameter override in production mode: lr={lr} vs config={cfg_lr}")
        if weight_decay is not None and weight_decay != cfg_wd:
            raise ValueError(f"Unapproved hyperparameter override in production mode: weight_decay={weight_decay} vs config={cfg_wd}")
        if grad_clip_norm is not None and grad_clip_norm != cfg_clip:
            raise ValueError(f"Unapproved hyperparameter override in production mode: grad_clip_norm={grad_clip_norm} vs config={cfg_clip}")
        if betas is not None and betas != cfg_betas:
            raise ValueError(f"Unapproved hyperparameter override in production mode: betas={betas} vs config={cfg_betas}")
        if eps is not None and eps != cfg_eps:
            raise ValueError(f"Unapproved hyperparameter override in production mode: eps={eps} vs config={cfg_eps}")
        if effective_batch_size is not None and effective_batch_size != cfg_eb:
            raise ValueError(f"Unapproved hyperparameter override in production mode: effective_batch_size={effective_batch_size} vs config={cfg_eb}")

    if lr is None:
        lr = cfg_lr
    if weight_decay is None:
        weight_decay = cfg_wd
    if grad_clip_norm is None:
        grad_clip_norm = cfg_clip
    if betas is None:
        betas = cfg_betas
    if eps is None:
        eps = cfg_eps
    if effective_batch_size is None:
        effective_batch_size = cfg_eb
    if min_epochs is None:
        min_epochs = int(n_cfg.get("min_epochs", 5))
    if max_epochs is None:
        max_epochs = int(n_cfg.get("max_epochs", 50))
    if patience is None:
        patience = int(n_cfg.get("early_stop_patience", 5))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    # Assert optimizer parameter groups strictly match approved configuration (C3)
    for pg in optimizer.param_groups:
        assert pg["lr"] == lr, f"Optimizer lr mismatch: {pg['lr']} vs {lr}"
        assert pg["weight_decay"] == weight_decay, f"Optimizer weight_decay mismatch: {pg['weight_decay']} vs {weight_decay}"
        assert pg["betas"] == betas, f"Optimizer betas mismatch: {pg['betas']} vs {betas}"
        assert pg["eps"] == eps, f"Optimizer eps mismatch: {pg['eps']} vs {eps}"

    selector = EarlyStoppingSelector(min_epochs=min_epochs, max_epochs=max_epochs, patience=patience)

    N_train = len(train_y)
    weights_np = compute_equal_market_weights(train_markets, target_num_markets=len(np.unique(train_markets)))
    train_w = torch.tensor(weights_np, dtype=torch.float32)

    unique_mkts, counts = np.unique(train_markets, return_counts=True)
    samples_per_market = {str(m): int(c) for m, c in zip(unique_mkts, counts)}

    macro_step = 0
    micro_step = 0
    epoch_val_losses: List[float] = []

    # Local RNG for dataset shuffling
    rng = np.random.default_rng(seed)

    start_epoch = 1
    initial_cursor = 0
    resumed_permutation = None
    if resume_from_checkpoint is not None:
        chk_file = Path(resume_from_checkpoint)
        if not chk_file.is_file() and chk_file.is_dir():
            chk_file = chk_file / "last_checkpoint.pt"
        if not chk_file.exists():
            raise FileNotFoundError(f"Requested resume checkpoint does not exist: {chk_file}")
        state = torch.load(chk_file, weights_only=False)
        start_epoch = restore_training_state(state, model, optimizer, selector, generator=rng)
        macro_step = state.macro_step
        micro_step = state.micro_step
        initial_cursor = getattr(state, "sampler_cursor", 0)
        if initial_cursor > 0 and getattr(state, "epoch_permutation", None) is not None:
            resumed_permutation = np.array(state.epoch_permutation, dtype=int)
        epoch_val_losses = list(getattr(state, "epoch_val_losses", []))
        # Validate restored optimizer parameter groups against current optimizer config (C3, C5)
        for pg in optimizer.param_groups:
            assert pg["lr"] == lr, f"Restored optimizer lr mismatch: {pg['lr']} vs {lr}"
            assert pg["weight_decay"] == weight_decay, f"Restored optimizer weight_decay mismatch: {pg['weight_decay']} vs {weight_decay}"
            assert pg["betas"] == betas, f"Restored optimizer betas mismatch: {pg['betas']} vs {betas}"
            assert pg["eps"] == eps, f"Restored optimizer eps mismatch: {pg['eps']} vs {eps}"
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if checkpoint_dir is not None:
        chk_path = Path(checkpoint_dir)
        chk_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        if epoch == start_epoch and resumed_permutation is not None:
            indices = resumed_permutation
            resumed_permutation = None
        else:
            indices = rng.permutation(N_train)

        # Macro-batch accumulation loop (supporting mid-epoch resume cursor)
        macro_start_offset = initial_cursor if (epoch == start_epoch and initial_cursor > 0) else 0
        initial_cursor = 0  # reset after first resumed epoch
        interrupted_mid_epoch = False

        for macro_start in range(macro_start_offset, N_train, effective_batch_size):
            macro_end = min(macro_start + effective_batch_size, N_train)
            macro_len = macro_end - macro_start
            if macro_len <= 0:
                continue

            optimizer.zero_grad()
            for micro_start in range(macro_start, macro_end, micro_batch_size):
                micro_end = min(micro_start + micro_batch_size, macro_end)
                batch_idx = indices[micro_start:micro_end]
                b_size = len(batch_idx)
                if b_size <= 0:
                    continue

                bx = train_x[batch_idx].to(device)
                by = train_y[batch_idx].to(device)
                bw = train_w[batch_idx].to(device)

                pred = model(bx)
                loss_unreduced = (pred - by) ** 2
                loss_weighted = torch.mean(loss_unreduced * bw)

                # Scale loss by actual micro-batch fraction of macro-batch
                loss_scaled = loss_weighted * (b_size / macro_len)
                loss_scaled.backward()
                micro_step += 1

            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            macro_step += 1

            # Safe macro-boundary interruption (C5)
            if interrupt_at_macro_step is not None and macro_step == interrupt_at_macro_step:
                if macro_end < N_train:
                    # Interrupted strictly mid-epoch before final macro step
                    if checkpoint_dir is not None:
                        interrupted_state = capture_training_state(
                            model=model,
                            optimizer=optimizer,
                            selector=selector,
                            epoch=epoch,
                            macro_step=macro_step,
                            micro_step=micro_step,
                            sampler_cursor=macro_end,
                            generator=rng,
                            epoch_val_losses=epoch_val_losses,
                            epoch_permutation=list(indices),
                        )
                        torch.save(interrupted_state, Path(checkpoint_dir) / "last_checkpoint.pt")
                    interrupted_mid_epoch = True
                    break
                else:
                    # Interrupted on final macro update of epoch:
                    # Execute validation and checkpoint selection before saving checkpoint and exiting (C5)
                    model.eval()
                    val_preds_list = []
                    with torch.no_grad():
                        for v_start in range(0, len(val_y), micro_batch_size):
                            v_end = min(v_start + micro_batch_size, len(val_y))
                            bx = val_x[v_start:v_end].to(device)
                            bp = model(bx).cpu().numpy()
                            val_preds_list.append(bp)
                        val_preds = np.concatenate(val_preds_list, axis=0) if val_preds_list else np.array([], dtype=np.float32)
                        val_targets = val_y.cpu().numpy()
                        val_loss = compute_equal_market_val_mse(val_preds, val_targets, val_markets)

                    epoch_val_losses.append(val_loss)
                    is_best = selector.step(epoch, val_loss)

                    if checkpoint_dir is not None:
                        state = capture_training_state(
                            model=model,
                            optimizer=optimizer,
                            selector=selector,
                            epoch=epoch,
                            macro_step=macro_step,
                            micro_step=micro_step,
                            sampler_cursor=0,
                            generator=rng,
                            epoch_val_losses=epoch_val_losses,
                        )
                        torch.save(state, Path(checkpoint_dir) / "last_checkpoint.pt")
                        if is_best:
                            torch.save(state, Path(checkpoint_dir) / "best_checkpoint.pt")

                    return model, TrainingSummary(
                        model_type=model.__class__.__name__,
                        seed=seed,
                        best_epoch=selector.best_epoch or 0,
                        best_loss=selector.best_loss if selector.best_loss is not None else float("inf"),
                        epochs_trained=len(epoch_val_losses),
                        total_macro_steps=macro_step,
                        samples_per_market=samples_per_market,
                        epoch_val_losses=epoch_val_losses,
                        early_stopped=False,
                    )

        if interrupted_mid_epoch:
            return model, TrainingSummary(
                model_type=model.__class__.__name__,
                seed=seed,
                best_epoch=selector.best_epoch or 0,
                best_loss=selector.best_loss if selector.best_loss is not None else float("inf"),
                epochs_trained=epoch,
                total_macro_steps=macro_step,
                samples_per_market=samples_per_market,
                epoch_val_losses=epoch_val_losses,
                early_stopped=False,
            )

        # Validation streamed in micro-batches
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for v_start in range(0, len(val_y), micro_batch_size):
                v_end = min(v_start + micro_batch_size, len(val_y))
                bx = val_x[v_start:v_end].to(device)
                bp = model(bx).cpu().numpy()
                val_preds_list.append(bp)
            val_preds = np.concatenate(val_preds_list, axis=0) if val_preds_list else np.array([], dtype=np.float32)
            val_targets = val_y.cpu().numpy()
            val_loss = compute_equal_market_val_mse(val_preds, val_targets, val_markets)

        epoch_val_losses.append(val_loss)
        is_best = selector.step(epoch, val_loss)

        # Save checkpoint if directory supplied
        if checkpoint_dir is not None:
            state = capture_training_state(
                model=model,
                optimizer=optimizer,
                selector=selector,
                epoch=epoch,
                macro_step=macro_step,
                micro_step=micro_step,
                sampler_cursor=0,
                generator=rng,
                epoch_val_losses=epoch_val_losses,
            )
            torch.save(state, Path(checkpoint_dir) / "last_checkpoint.pt")
            if is_best:
                torch.save(state, Path(checkpoint_dir) / "best_checkpoint.pt")

        if interrupt_at_epoch is not None and epoch == interrupt_at_epoch:
            break

        if selector.should_stop:
            break

    # Load best weights before returning
    if checkpoint_dir is not None and (Path(checkpoint_dir) / "best_checkpoint.pt").exists():
        best_state = torch.load(Path(checkpoint_dir) / "best_checkpoint.pt", weights_only=False)
        model.load_state_dict(best_state.model_state)

    summary = TrainingSummary(
        model_type=model.__class__.__name__,
        seed=seed,
        best_epoch=selector.best_epoch,
        best_loss=selector.best_loss,
        epochs_trained=len(epoch_val_losses),
        total_macro_steps=macro_step,
        samples_per_market=samples_per_market,
        epoch_val_losses=epoch_val_losses,
        early_stopped=selector.should_stop,
    )
    return model, summary
