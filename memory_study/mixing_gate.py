"""
Selective Trust Gate Module for Round 5.

Implements:
1. G_base: Backbone alone (lambda=0)
2. G_fixed: Fixed mixing (lambda=0.25)
3. G_grid: Grid-tuned fixed lambda* in {0.0, 0.1, 0.25, 0.5, 1.0} on H2 2021
4. G_gate: Learned linear logistic trust gate g_t = sigmoid(a^T u_t + b)
5. G_mem: Memory alone (lambda=1.0)
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path("c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent")
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = PROJECT_ROOT / "research_runs" / "memory_study" / "cache"


class SelectiveTrustGate(nn.Module):
    """Linear logistic trust gate g_t = sigmoid(a^T u_t + b)."""
    def __init__(self, in_features: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        # Initialize bias so initial gate is ~0.25: log(0.25 / 0.75) = -1.0986
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, -1.0986)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, 4)
        return torch.sigmoid(self.linear(u)).squeeze(-1)


def fit_trust_gate(
    backbone: str,
    backbone_seed: int,
    base_preds_h2: np.ndarray,
    mem_preds_h2: np.ndarray,
    vols_h2: np.ndarray,
    y_true_h2: np.ndarray,
    epochs: int = 50,
    lr: float = 0.05,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[SelectiveTrustGate, float]:
    """
    Fits trust gate and finds optimal grid lambda* on H2 2021 validation set.
    """
    ckpt_path = MODELS_DIR / f"trust_gate_{backbone.lower()}_seed_{backbone_seed}.pt"

    # 1. Grid search fixed lambda
    grid_lambdas = [0.0, 0.10, 0.25, 0.50, 1.0]
    best_grid_lam = 0.25
    best_grid_mse = 999.0
    for lam in grid_lambdas:
        hyb = (1.0 - lam) * base_preds_h2 + lam * mem_preds_h2
        mse = float(np.mean((y_true_h2 - hyb) ** 2))
        if mse < best_grid_mse:
            best_grid_mse = mse
            best_grid_lam = lam

    # 2. Build gate feature matrix u_t: [|base|, |base - mem|, mem_val, vol]
    u_mat = np.stack([
        np.abs(base_preds_h2),
        np.abs(base_preds_h2 - mem_preds_h2),
        np.abs(mem_preds_h2),
        vols_h2,
    ], axis=1).astype(np.float32)

    # Standardize features
    u_mean = np.mean(u_mat, axis=0, keepdims=True)
    u_std = np.std(u_mat, axis=0, keepdims=True) + 1e-4
    u_norm = (u_mat - u_mean) / u_std

    u_t = torch.tensor(u_norm, dtype=torch.float32, device=device)
    base_t = torch.tensor(base_preds_h2, dtype=torch.float32, device=device)
    mem_t = torch.tensor(mem_preds_h2, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_true_h2, dtype=torch.float32, device=device)

    model = SelectiveTrustGate(in_features=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    for ep in range(epochs):
        model.train()
        g = model(u_t)
        hyb = (1.0 - g) * base_t + g * mem_t
        loss = F.mse_loss(hyb, y_t)

        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    torch.save({
        "state_dict": model.state_dict(),
        "u_mean": torch.tensor(u_mean, dtype=torch.float32),
        "u_std": torch.tensor(u_std, dtype=torch.float32),
        "best_grid_lam": float(best_grid_lam),
    }, ckpt_path)

    print(f"   [+] {backbone.upper()} Seed {backbone_seed} Trust Gate fitted. Best Grid Lambda: {best_grid_lam} (MSE: {best_grid_mse:.6f})")
    return model, best_grid_lam


def load_trust_gate(
    backbone: str,
    backbone_seed: int,
    device: torch.device = torch.device("cpu"),
) -> Tuple[SelectiveTrustGate, np.ndarray, np.ndarray, float]:
    """Loads fitted trust gate, feature normalizers, and best grid lambda."""
    ckpt_path = MODELS_DIR / f"trust_gate_{backbone.lower()}_seed_{backbone_seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Trust gate checkpoint not found at {ckpt_path}")
    data = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SelectiveTrustGate(in_features=4).to(device)
    model.load_state_dict(data["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    u_m = data["u_mean"]
    u_s = data["u_std"]
    if isinstance(u_m, torch.Tensor):
        u_m = u_m.cpu().numpy()
    if isinstance(u_s, torch.Tensor):
        u_s = u_s.cpu().numpy()
    return model, u_m, u_s, float(data["best_grid_lam"])

