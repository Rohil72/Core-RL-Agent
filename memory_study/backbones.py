"""
Frozen Backbone Model Loaders and Shared Representation Interfaces.

Supports:
1. AnnualPatchTemporalTransformer (Canonical C09 Transformer)
2. MLPEncoder (Matched 2-Layer MLP Comparator)

Both backbones expose:
- Decimal return predictions for 63-session horizon
- 128-dimensional latent representations
- Strict parameter freezing (eval mode, requires_grad=False)
"""

from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path("c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent")
MODELS_DIR = PROJECT_ROOT / "exports" / "CORE_RL_V4_VERIFIED_GOVERNANCE_PACKAGE" / "models"
LOCAL_MODELS_DIR = PROJECT_ROOT / "models"
LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)


class AnnualPatchTemporalTransformer(nn.Module):
    """Architecture used by canonical C09 replay checkpoints."""
    def __init__(self, input_dim: int = 23, embed_dim: int = 64, num_heads: int = 4, latent_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=128, batch_first=True, dropout=0.1, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.Linear(embed_dim, 1)
        self.latent_head = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.outcome_head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.input_proj(x)
        T = h.size(1)
        if T > 1:
            pos = torch.arange(T, device=h.device, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.embed_dim, 2, device=h.device, dtype=torch.float32)
                * (-2.302585092994046 * 4 / self.embed_dim)
            )
            pe = torch.zeros(T, self.embed_dim, device=h.device)
            pe[:, 0::2] = torch.sin(pos * div_term)
            pe[:, 1::2] = torch.cos(pos * div_term)
            h = h + pe.unsqueeze(0)

        h_trans = self.transformer(h)
        weights = torch.softmax(self.pool(h_trans), dim=1)
        h_pool = torch.sum(h_trans * weights, dim=1)

        latent = self.latent_head(h_pool)
        pred_outcome = self.outcome_head(latent)
        return latent, pred_outcome


class MLPEncoder(nn.Module):
    """2-Layer MLP Comparator for Exp 14 and Memory Study."""
    def __init__(self, input_dim: int = 23, hidden_dim: int = 64, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            # If passed patch sequence (B, T, D), take the last step feature
            x = x[:, -1, :]
        lat = self.net(x)
        pred = self.head(lat)
        return lat, pred


def freeze_model(model: nn.Module) -> nn.Module:
    """Strictly freezes model parameters and puts module in evaluation mode."""
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def verify_backbone_frozen(model: nn.Module) -> bool:
    """Acceptance check ensuring all parameters have requires_grad=False."""
    return all(not p.requires_grad for p in model.parameters()) and not model.training


def load_transformer_checkpoint(seed: int, device: torch.device) -> AnnualPatchTemporalTransformer:
    """Loads frozen C09 AnnualPatchTemporalTransformer for the given seed."""
    ckpt_path = MODELS_DIR / f"v4_metric_transformer_seed_{seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Transformer checkpoint not found at {ckpt_path}")
    model = AnnualPatchTemporalTransformer(input_dim=23, embed_dim=64, num_heads=4, latent_dim=128).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return freeze_model(model)


def load_mlp_checkpoint(seed: int, device: torch.device) -> MLPEncoder:
    """Loads frozen 2-layer MLP for the given seed."""
    ckpt_path = LOCAL_MODELS_DIR / f"mlp_encoder_seed_{seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"MLP checkpoint not found at {ckpt_path}. Run training first.")
    model = MLPEncoder(input_dim=23, hidden_dim=64, latent_dim=128).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return freeze_model(model)


def ensure_mlp_checkpoints(train_x: np.ndarray, train_y: np.ndarray, seeds: list[int], device: torch.device):
    """
    Trains and saves MLP checkpoints on pre-2021 data if they don't already exist.
    Guarantees reproducible, persistent checkpoints on disk.
    """
    X_t = torch.tensor(train_x, dtype=torch.float32, device=device)
    y_t = torch.tensor(train_y, dtype=torch.float32, device=device).unsqueeze(1)
    ds = torch.utils.data.TensorDataset(X_t, y_t)

    for s in seeds:
        ckpt_path = LOCAL_MODELS_DIR / f"mlp_encoder_seed_{s}.pt"
        if ckpt_path.exists():
            continue
        print(f"   [+] Fitting persistent MLP checkpoint for seed {s}...")
        torch.manual_seed(s)
        mlp = MLPEncoder().to(device)
        opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.MSELoss()
        dl = torch.utils.data.DataLoader(ds, batch_size=4096, shuffle=True)

        mlp.train()
        for epoch in range(5):
            for b_x, b_y in dl:
                opt.zero_grad()
                _, p = mlp(b_x)
                loss = crit(p, b_y)
                loss.backward()
                opt.step()
        mlp.eval()
        torch.save(mlp.state_dict(), ckpt_path)
        print(f"   [+] Saved MLP checkpoint: {ckpt_path}")


def load_backbone(backbone_id: str, seed: int, device: torch.device) -> nn.Module:
    """Factory function to load frozen backbone by identifier and seed."""
    if backbone_id.lower() in ("transformer", "trans"):
        return load_transformer_checkpoint(seed, device)
    elif backbone_id.lower() in ("mlp", "mlp_encoder"):
        return load_mlp_checkpoint(seed, device)
    else:
        raise ValueError(f"Unknown backbone_id: {backbone_id}")
