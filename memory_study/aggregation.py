"""
Learned Aggregation Module for Round 3.

Implements:
1. A0: Fixed Softmax Kernel (tau=0.08)
2. A1: Learned Logit Adjustment:
       ell_i = s_i / tau + h_psi(u_i)
       where u_i = [s_i, dist_i, log_age_i]
       w_i = softmax(ell_i)
       r_mem = sum_i w_i r_i
3. A_ctrl: Query-Only Adapter Control (parameter-capacity matched comparator)
       Takes only query latent z_q (no neighbours) to isolate capacity effects.
"""

import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path("c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent")
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = PROJECT_ROOT / "research_runs" / "memory_study" / "cache"


class NeighborLogitAdjustment(nn.Module):
    """Query-conditioned logit adjustment network h_psi(u_i)."""
    def __init__(self, in_features: int = 3, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize output layer near zero so initial logits match fixed kernel exactly
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, u: torch.Tensor, base_sims: torch.Tensor, tau: float = 0.08) -> Tuple[torch.Tensor, torch.Tensor]:
        # u: (B, K, 3), base_sims: (B, K)
        delta_ell = self.net(u).squeeze(-1)  # (B, K)
        ell = base_sims / tau + delta_ell
        weights = torch.softmax(ell - torch.max(ell, dim=-1, keepdim=True).values, dim=-1)
        return weights, ell


class QueryOnlyAdapter(nn.Module):
    """Query-only capacity control network f_theta(z_q)."""
    def __init__(self, in_dim: int = 128, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        # z_q: (B, in_dim)
        return self.net(z_q).squeeze(-1)


def train_aggregation_models(
    backbone: str,
    backbone_seed: int,
    epochs: int = 5,
    lr: float = 1e-3,
    tau: float = 0.08,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[NeighborLogitAdjustment, QueryOnlyAdapter]:
    """
    Trains A1 (Learned Logit Adjustment) and A_ctrl (Query-Only Adapter) on development period.
    Saves checkpoints and returns trained modules.
    """
    a1_path = MODELS_DIR / f"agg_a1_{backbone.lower()}_seed_{backbone_seed}.pt"
    ctrl_path = MODELS_DIR / f"agg_ctrl_{backbone.lower()}_seed_{backbone_seed}.pt"

    print(f"\n--- Training Round 3 Aggregation Models for {backbone.upper()} (Seed {backbone_seed}) on {device} ---")

    # Load caches
    mm = pd.read_parquet(CACHE_DIR / "memory_meta.parquet")
    lats = np.load(CACHE_DIR / f"memory_latents_{backbone.lower()}_seed_{backbone_seed}.npy")

    train_mask = (mm["origin_timestamp"] >= "2018-01-01") & (mm["origin_timestamp"] <= "2019-12-31")
    val_mask = (mm["origin_timestamp"] >= "2020-01-01") & (mm["origin_timestamp"] <= "2020-07-07")
    cand_mask = mm["available_timestamp"] <= "2017-12-31"

    train_idx = mm[train_mask].index.values
    val_idx = mm[val_mask].index.values
    cand_idx = mm[cand_mask].index.values

    # Pre-center
    m_cand_raw = torch.tensor(lats[cand_idx], dtype=torch.float32, device=device)
    m_mean = torch.mean(m_cand_raw, dim=0, keepdim=True)

    q_train = torch.tensor(lats[train_idx], dtype=torch.float32, device=device) - m_mean
    y_train = torch.tensor(mm.loc[train_idx, "return_63"].values, dtype=torch.float32, device=device)
    tkr_train = mm.loc[train_idx, "ticker"].values

    q_val = torch.tensor(lats[val_idx], dtype=torch.float32, device=device) - m_mean
    y_val = torch.tensor(mm.loc[val_idx, "return_63"].values, dtype=torch.float32, device=device)
    tkr_val = mm.loc[val_idx, "ticker"].values

    m_cand = m_cand_raw - m_mean
    y_cand = torch.tensor(mm.loc[cand_idx, "return_63"].values, dtype=torch.float32, device=device)
    tkr_cand = mm.loc[cand_idx, "ticker"].values

    # Normalized keys
    k_q_train = F.normalize(q_train, p=2, dim=-1)
    k_q_val = F.normalize(q_val, p=2, dim=-1)
    k_m_cand = F.normalize(m_cand, p=2, dim=-1)

    # 1. Train A1 (Neighbor Logit Adjustment)
    model_a1 = NeighborLogitAdjustment(in_features=3, hidden_dim=16).to(device)
    opt_a1 = torch.optim.Adam(model_a1.parameters(), lr=lr)

    # Pre-select top-25 neighbours for a fixed validation sample (1,000 queries)
    rng = np.random.default_rng(42)
    val_sub = rng.choice(len(q_val), size=1000, replace=False)
    cand_sub = rng.choice(len(m_cand), size=8000, replace=False)

    q_val_s = k_q_val[val_sub]
    y_val_s = y_val[val_sub]
    m_cand_s = k_m_cand[cand_sub]
    y_cand_s = y_cand[cand_sub]

    val_sims = q_val_s @ m_cand_s.T
    val_mask_same = torch.tensor((tkr_val[val_sub, None] == tkr_cand[cand_sub, None].T), dtype=torch.bool, device=device)
    val_sims = val_sims.masked_fill(val_mask_same, -1e9)
    val_topk_s, val_topk_i = torch.topk(val_sims, k=25, dim=1)
    val_topk_y = y_cand_s[val_topk_i]

    # Compute u for validation: u = [s, dist=2*(1-s), log_age=1.0 placeholder]
    val_u = torch.stack([
        val_topk_s,
        torch.clamp(2.0 * (1.0 - val_topk_s), min=0.0),
        torch.ones_like(val_topk_s) * 2.5,
    ], dim=-1)

    # Baseline A0 MSE
    w0 = F.softmax(val_topk_s / tau, dim=-1)
    a0_pred = torch.sum(w0 * val_topk_y, dim=-1)
    a0_mse = F.mse_loss(a0_pred, y_val_s).item()
    print(f"   [A0 Baseline] Val MSE: {a0_mse:.6f}")

    best_a1_mse = a0_mse
    best_a1_weights = {k: v.cpu().clone() for k, v in model_a1.state_dict().items()}

    for ep in range(1, epochs + 1):
        model_a1.train()
        perm = torch.randperm(len(q_train))
        ep_loss = 0.0
        n_b = 0

        for i in range(0, min(len(q_train), 15360), 512):
            b_idx = perm[i:i+512]
            b_q = k_q_train[b_idx]
            b_y = y_train[b_idx]
            b_tkr = tkr_train[b_idx.cpu().numpy()]

            c_perm = torch.randperm(len(m_cand))[:4096]
            b_m = k_m_cand[c_perm]
            b_my = y_cand[c_perm]
            b_mtkr = tkr_cand[c_perm.cpu().numpy()]

            same_t = torch.tensor((b_tkr[:, None] == b_mtkr[None, :]), dtype=torch.bool, device=device)
            sims = (b_q @ b_m.T).masked_fill(same_t, -1e9)
            topk_s, topk_i = torch.topk(sims, k=25, dim=1)
            topk_y = b_my[topk_i]

            u = torch.stack([
                topk_s,
                torch.clamp(2.0 * (1.0 - topk_s), min=0.0),
                torch.ones_like(topk_s) * 2.5,
            ], dim=-1)

            w_pred, _ = model_a1(u, topk_s, tau=tau)
            pred = torch.sum(w_pred * topk_y, dim=-1)
            loss = F.mse_loss(pred, b_y)

            opt_a1.zero_grad()
            loss.backward()
            opt_a1.step()

            ep_loss += loss.item()
            n_b += 1

        model_a1.eval()
        with torch.no_grad():
            w_val, _ = model_a1(val_u, val_topk_s, tau=tau)
            val_pred = torch.sum(w_val * val_topk_y, dim=-1)
            val_mse = F.mse_loss(val_pred, y_val_s).item()

        if val_mse < best_a1_mse:
            best_a1_mse = val_mse
            best_a1_weights = {k: v.cpu().clone() for k, v in model_a1.state_dict().items()}
        print(f"   [Epoch {ep}/{epochs}] A1 Train Loss: {ep_loss/n_b:.6f} | Val MSE: {val_mse:.6f}")

    model_a1.load_state_dict({k: v.to(device) for k, v in best_a1_weights.items()})
    for p in model_a1.parameters():
        p.requires_grad = False
    model_a1.eval()
    torch.save(model_a1.state_dict(), a1_path)

    # 2. Train A_ctrl (Query-Only Adapter)
    model_ctrl = QueryOnlyAdapter(in_dim=128, hidden_dim=16).to(device)
    opt_ctrl = torch.optim.Adam(model_ctrl.parameters(), lr=lr)

    best_ctrl_mse = 999.0
    best_ctrl_weights = {k: v.cpu().clone() for k, v in model_ctrl.state_dict().items()}

    for ep in range(1, epochs + 1):
        model_ctrl.train()
        perm = torch.randperm(len(q_train))
        ep_loss = 0.0
        n_b = 0

        for i in range(0, min(len(q_train), 15360), 512):
            b_idx = perm[i:i+512]
            b_z = q_train[b_idx]
            b_y = y_train[b_idx]

            pred_adj = model_ctrl(b_z)
            loss = F.mse_loss(pred_adj, b_y)

            opt_ctrl.zero_grad()
            loss.backward()
            opt_ctrl.step()

            ep_loss += loss.item()
            n_b += 1

        model_ctrl.eval()
        with torch.no_grad():
            val_pred_ctrl = model_ctrl(q_val[val_sub])
            val_mse_ctrl = F.mse_loss(val_pred_ctrl, y_val_s).item()

        if val_mse_ctrl < best_ctrl_mse:
            best_ctrl_mse = val_mse_ctrl
            best_ctrl_weights = {k: v.cpu().clone() for k, v in model_ctrl.state_dict().items()}

    model_ctrl.load_state_dict({k: v.to(device) for k, v in best_ctrl_weights.items()})
    for p in model_ctrl.parameters():
        p.requires_grad = False
    model_ctrl.eval()
    torch.save(model_ctrl.state_dict(), ctrl_path)

    print(f"   [+] Saved A1 checkpoint ({a1_path.name}) and A_ctrl checkpoint ({ctrl_path.name})")
    return model_a1, model_ctrl
