"""
Frozen Backbone Model Loaders and Shared Representation Interfaces.

Supports:
1. AnnualPatchTemporalTransformer (Canonical C09 Transformer)
2. MLPEncoder (Matched 2-Layer MLP Comparator)

Both backbones expose:
- Decimal return predictions for 63-session horizon
- 128-dimensional latent representations
- Strict parameter freezing (eval mode, requires_grad=False)
- Representation compatibility verification
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_study.shared_representation import MLP_REPRESENTATION_ID

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
    """
    2-Layer MLP Comparator for Exp 14 and Memory Study.
    Explicitly requires 2D input (batch_size, 23).
    Rejects 3D inputs to prevent silent mismatch.
    """
    def __init__(self, input_dim: int = 23, hidden_dim: int = 64, latent_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"MLPEncoder explicitly requires 2D input tensor of shape (batch_size, {self.input_dim}), "
                f"got shape {tuple(x.shape)}. 3D patch sequences must be pooled explicitly prior to forward()."
            )
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


def load_mlp_checkpoint(seed: int, device: torch.device, expected_rep_id: str = MLP_REPRESENTATION_ID) -> MLPEncoder:
    """
    Loads frozen 2-layer MLP for the given seed.
    Strictly verifies representation compatibility against expected_rep_id.
    """
    ckpt_path = LOCAL_MODELS_DIR / f"mlp_encoder_seed_{seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"MLP checkpoint not found at {ckpt_path}. Run training first.")

    loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(loaded, dict) and "representation_id" in loaded:
        rep_id = loaded.get("representation_id")
        if expected_rep_id is not None and rep_id != expected_rep_id:
            raise RuntimeError(
                f"Artifact compatibility rejection: checkpoint {ckpt_path.name} has representation_id '{rep_id}', "
                f"expected '{expected_rep_id}'. Run training to regenerate corrected weights."
            )
        state_dict = loaded["state_dict"]
    elif isinstance(loaded, dict) and "net.0.weight" in loaded:
        # Legacy untagged checkpoint without representation_id
        if expected_rep_id is not None:
            raise RuntimeError(
                f"Artifact compatibility rejection: checkpoint {ckpt_path.name} is a legacy untagged checkpoint "
                f"without representation_id metadata. Expected '{expected_rep_id}'."
            )
        state_dict = loaded
    else:
        raise ValueError(f"Unknown checkpoint format at {ckpt_path}")

    model = MLPEncoder(input_dim=23, hidden_dim=64, latent_dim=128).to(device)
    model.load_state_dict(state_dict)
    return freeze_model(model)


def ensure_mlp_checkpoints(
    train_x: np.ndarray,
    train_y: np.ndarray,
    seeds: List[int],
    device: torch.device,
    force_retrain: bool = False,
    representation_id: str = MLP_REPRESENTATION_ID
) -> Dict[int, Dict[str, Any]]:
    """
    Trains and saves MLP checkpoints on pre-2021 data.
    Guarantees reproducible, persistent checkpoints on disk with full provenance logging.
    """
    if train_x.ndim != 2 or train_x.shape[1] != 23:
        raise ValueError(f"train_x must be 2D array of shape (N, 23), got {train_x.shape}")
    if len(train_x) != len(train_y):
        raise ValueError(f"train_x count ({len(train_x)}) != train_y count ({len(train_y)})")

    X_t = torch.tensor(train_x, dtype=torch.float32, device=device)
    y_t = torch.tensor(train_y, dtype=torch.float32, device=device).unsqueeze(1)
    ds = torch.utils.data.TensorDataset(X_t, y_t)

    input_hash = hashlib.sha256(train_x.tobytes()).hexdigest()
    target_hash = hashlib.sha256(train_y.tobytes()).hexdigest()
    train_records = {}

    for s in seeds:
        ckpt_path = LOCAL_MODELS_DIR / f"mlp_encoder_seed_{s}.pt"
        if not force_retrain and ckpt_path.exists():
            try:
                loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(loaded, dict) and loaded.get("representation_id") == representation_id:
                    print(f"   [+] Verified compatible MLP checkpoint for seed {s} ({representation_id})")
                    continue
            except Exception:
                pass

        print(f"   [+] Fitting corrected MLP checkpoint for seed {s} (representation={representation_id})...")
        torch.manual_seed(s)
        mlp = MLPEncoder(input_dim=23, hidden_dim=64, latent_dim=128).to(device)
        opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.MSELoss()
        dl = torch.utils.data.DataLoader(ds, batch_size=4096, shuffle=True)

        mlp.train()
        loss_history = []
        for epoch in range(5):
            ep_loss = 0.0
            n_b = 0
            for b_x, b_y in dl:
                opt.zero_grad()
                _, p = mlp(b_x)
                loss = crit(p, b_y)
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
                n_b += 1
            avg_loss = ep_loss / max(n_b, 1)
            loss_history.append(avg_loss)
            print(f"       Epoch {epoch+1}/5 Loss: {avg_loss:.6f}")

        mlp.eval()
        save_dict = {
            "representation_id": representation_id,
            "architecture": "MLPEncoder_2Layer_GELU_23x64x128x1",
            "seed": s,
            "state_dict": mlp.state_dict(),
            "train_config": {
                "epochs": 5,
                "batch_size": 4096,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "loss": "MSELoss",
                "optimizer": "AdamW",
                "seed": s
            },
            "provenance": {
                "n_samples": len(train_x),
                "input_dim": 23,
                "input_sha256": input_hash,
                "target_sha256": target_hash,
                "loss_history": loss_history,
                "final_loss": loss_history[-1] if loss_history else None,
                "trained_at_utc": datetime.now(timezone.utc).isoformat()
            }
        }
        torch.save(save_dict, ckpt_path)
        print(f"   [+] Saved corrected MLP checkpoint: {ckpt_path} (Final loss: {loss_history[-1]:.6f})")
        train_records[s] = save_dict["provenance"]

    return train_records


def load_backbone(backbone_id: str, seed: int, device: torch.device) -> nn.Module:
    """Factory function to load frozen backbone by identifier and seed."""
    if backbone_id.lower() in ("transformer", "trans"):
        return load_transformer_checkpoint(seed, device)
    elif backbone_id.lower() == "mlp":
        return load_mlp_checkpoint(seed, device)
    else:
        raise ValueError(f"Unknown backbone_id: {backbone_id}")
