"""
Key Projection Module for Round 2: Learned Retrieval Keys.

Defines:
- KeyProjection: 32-dimensional linear projection g_phi(z) = normalize(W z)
- K0: Identity projection on original 128-d normalized latents
- K1: Frozen random 32-d projection (seeded reproducibly)
- K2: Trainable 32-d linear projection initialized from K1 and fitted on
      development-period memory forecast MSE with same-ticker exclusion.
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


class KeyProjection(nn.Module):
    """32-dimensional linear metric projection onto the unit sphere S^31."""
    def __init__(self, in_dim: int = 128, out_dim: int = 32, init_seed: int = 42):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.reset_parameters(init_seed)

    def reset_parameters(self, seed: int):
        gen = torch.Generator().manual_seed(seed)
        # Standard Gaussian projection scaled by 1/sqrt(out_dim)
        w = torch.randn(self.out_dim, self.in_dim, generator=gen) / np.sqrt(self.out_dim)
        self.proj.weight.data.copy_(w)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (..., in_dim)
        h = self.proj(z)
        return F.normalize(h, p=2, dim=-1)


def get_k1_random_projection(in_dim: int = 128, out_dim: int = 32, seed: int = 42, device: torch.device = torch.device("cpu")) -> KeyProjection:
    """Returns frozen random 32-d projection K1."""
    model = KeyProjection(in_dim=in_dim, out_dim=out_dim, init_seed=seed).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def train_k2_projection(
    backbone: str,
    backbone_seed: int,
    init_seed: int = 42,
    lr: float = 1e-2,
    epochs: int = 6,
    batch_size: int = 512,
    cand_sample_size: int = 4096,
    tau: float = 0.08,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> KeyProjection:
    """
    Trains 32-d projection W to minimize development-period memory forecast MSE:
    L_key = (1/B) sum_q ( r_q^63 - sum_i w_i r_i^63 )^2
    with strict temporal maturity (pre-2018 candidates) and same-ticker entity exclusion.
    Saves and returns the best checkpoint based on 2020 validation MSE.
    """
    ckpt_path = MODELS_DIR / f"key_proj_{backbone.lower()}_seed_{backbone_seed}.pt"

    print(f"\n--- Training K2 Projection for {backbone.upper()} (Seed {backbone_seed}) on {device} ---")
    mm = pd.read_parquet(CACHE_DIR / "memory_meta.parquet")
    lats = np.load(CACHE_DIR / f"memory_latents_{backbone.lower()}_seed_{backbone_seed}.npy")

    # Chronological partition of development memory:
    # Candidates: available <= 2017-12-31 (87,733 records)
    # Training Queries: 2018-01-01 to 2019-12-31 (51,285 records)
    # Validation Queries: 2020-01-01 to 2020-07-07 (12,875 records)
    train_mask = (mm["origin_timestamp"] >= "2018-01-01") & (mm["origin_timestamp"] <= "2019-12-31")
    val_mask = (mm["origin_timestamp"] >= "2020-01-01") & (mm["origin_timestamp"] <= "2020-07-07")
    cand_mask = mm["available_timestamp"] <= "2017-12-31"

    train_idx = mm[train_mask].index.values
    val_idx = mm[val_mask].index.values
    cand_idx = mm[cand_mask].index.values

    # Pre-center by candidate pool mean to remove common-mode anisotropy
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

    # Initialize K2 from K1 weights
    model = KeyProjection(in_dim=128, out_dim=32, init_seed=init_seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Validation evaluation helper
    # Subsample 1,000 diverse validation queries for fast epoch tracking
    rng = np.random.default_rng(123)
    val_sub_idx = rng.choice(len(q_val), size=1500, replace=False)
    q_val_sub = q_val[val_sub_idx]
    y_val_sub = y_val[val_sub_idx]
    tkr_val_sub = tkr_val[val_sub_idx]

    cand_sub_idx = rng.choice(len(m_cand), size=8000, replace=False)
    m_cand_val = m_cand[cand_sub_idx]
    y_cand_val = y_cand[cand_sub_idx]
    tkr_cand_val = tkr_cand[cand_sub_idx]

    # Precompute validation same-ticker mask: shape (1500, 8000)
    val_mask_same = torch.tensor((tkr_val_sub[:, None] == tkr_cand_val[None, :]), dtype=torch.bool, device=device)

    def eval_val_mse(mod: nn.Module) -> float:
        mod.eval()
        with torch.no_grad():
            k_q = mod(q_val_sub)
            k_m = mod(m_cand_val)
            sims = k_q @ k_m.T
            sims = sims.masked_fill(val_mask_same, -1e9)
            topk_sims, topk_i = torch.topk(sims, k=25, dim=1)
            w = F.softmax(topk_sims / tau, dim=1)
            pred = torch.sum(w * y_cand_val[topk_i], dim=1)
            mse = F.mse_loss(pred, y_val_sub).item()
        return mse

    best_val_mse = eval_val_mse(model)
    best_weights = model.proj.weight.data.clone()
    print(f"   [Epoch 0 / K1 Baseline] Val MSE: {best_val_mse:.6f}")

    n_train = len(q_train)
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        ep_loss = 0.0
        n_batches = 0

        # Steps per epoch: sample 15,000 queries per epoch across batches
        for i in range(0, min(n_train, 15360), batch_size):
            b_q_idx = perm[i:i + batch_size]
            b_q = q_train[b_q_idx]
            b_y = y_train[b_q_idx]
            b_tkr = tkr_train[b_q_idx.cpu().numpy()]

            # Sample candidate pool
            c_perm = torch.randperm(len(m_cand))[:cand_sample_size]
            b_m = m_cand[c_perm]
            b_my = y_cand[c_perm]
            b_mtkr = tkr_cand[c_perm.cpu().numpy()]

            # Same-ticker mask
            same_tkr = torch.tensor((b_tkr[:, None] == b_mtkr[None, :]), dtype=torch.bool, device=device)

            k_q = model(b_q)
            k_m = model(b_m)
            sims = k_q @ k_m.T
            sims = sims.masked_fill(same_tkr, -1e9)

            topk_sims, topk_i = torch.topk(sims, k=25, dim=1)
            w = F.softmax(topk_sims / tau, dim=1)
            pred = torch.sum(w * b_my[topk_i], dim=1)
            loss = F.mse_loss(pred, b_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_loss += loss.item()
            n_batches += 1

        val_mse = eval_val_mse(model)
        improved = val_mse < best_val_mse
        if improved:
            best_val_mse = val_mse
            best_weights = model.proj.weight.data.clone()

        status = "*" if improved else ""
        print(f"   [Epoch {ep}/{epochs}] Train Loss: {ep_loss / n_batches:.6f} | Val MSE: {val_mse:.6f} {status}")

    # Load best weights and save checkpoint
    model.proj.weight.data.copy_(best_weights)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    torch.save({
        "weight": model.proj.weight.data.cpu(),
        "m_mean": m_mean.cpu(),
        "best_val_mse": best_val_mse,
        "backbone": backbone,
        "backbone_seed": backbone_seed,
    }, ckpt_path)
    print(f"   [+] Saved best K2 checkpoint to {ckpt_path} (Val MSE: {best_val_mse:.6f}) in {time.time() - t0:.2f}s")
    return model


def load_k2_projection(backbone: str, backbone_seed: int, device: torch.device = torch.device("cpu")) -> Tuple[KeyProjection, torch.Tensor]:
    """Loads trained K2 projection and pre-2021 centering mean."""
    ckpt_path = MODELS_DIR / f"key_proj_{backbone.lower()}_seed_{backbone_seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"K2 checkpoint not found at {ckpt_path}. Train it first.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = KeyProjection(in_dim=128, out_dim=32).to(device)
    model.proj.weight.data.copy_(ckpt["weight"].to(device))
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    m_mean = ckpt["m_mean"].to(device)
    return model, m_mean
