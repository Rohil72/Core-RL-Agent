"""Neural backbones: MLP and Transformer with explicit seeded initialization (v2).

Architectures:
- MLP: Linear(966, 64) -> LayerNorm(64, eps=1e-5) -> GELU -> Linear(64, 128) -> LayerNorm(128, eps=1e-5) -> Linear(128, 1)
- Transformer: Linear(23, 64) -> sinusoidal pos enc -> 2 TransformerEncoderLayers (post-norm, 4 heads, d_ff=128, gelu, dropout=0.1) -> learned attention pool -> Linear(64, 128) -> LayerNorm(128, eps=1e-5) -> Linear(128, 1)
- Explicit initialization: Xavier-uniform on Linear weights, zero biases, LayerNorm weight=1/bias=0.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights_xavier(module: nn.Module, generator: Optional[torch.Generator] = None) -> None:
    """Explicit deterministic initialization: Xavier-uniform weights, zero biases."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, generator=generator)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with base 10000 for 42 sequence tokens."""

    def __init__(self, seq_len: int = 42, d_model: int = 64):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape (1, 42, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class LearnedAttentionPooling(nn.Module):
    """Learned scalar attention pooling over sequence tokens."""

    def __init__(self, d_model: int = 64, generator: Optional[torch.Generator] = None):
        super().__init__()
        self.attn_proj = nn.Linear(d_model, 1, bias=False)
        init_weights_xavier(self.attn_proj, generator=generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 42, 64)
        attn_logits = self.attn_proj(x)  # (B, 42, 1)
        weights = F.softmax(attn_logits, dim=1)  # (B, 42, 1)
        pooled = torch.sum(x * weights, dim=1)  # (B, 64)
        return pooled


class MLPAnnual(nn.Module):
    """966-d MLP: Linear(966, 64) -> LayerNorm(64) -> GELU -> Linear(64, 128) -> LayerNorm(128) -> Linear(128, 1)."""

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        self.fc1 = nn.Linear(966, 64)
        self.ln1 = nn.LayerNorm(64, eps=1e-5)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(64, 128)
        self.ln2 = nn.LayerNorm(128, eps=1e-5)
        self.head = nn.Linear(128, 1)

        # Apply explicit initialization
        init_weights_xavier(self.fc1, generator)
        init_weights_xavier(self.ln1, generator)
        init_weights_xavier(self.fc2, generator)
        init_weights_xavier(self.ln2, generator)
        init_weights_xavier(self.head, generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 966)
        if x.dim() != 2 or x.size(1) != 966:
            raise ValueError(f"MLPAnnual requires input shape (B, 966), got {x.shape}")
        h = self.act(self.ln1(self.fc1(x)))
        h = self.ln2(self.fc2(h))
        out = self.head(h).squeeze(-1)  # (B,)
        return out


class TransformerAnnual(nn.Module):
    """42x23 Transformer: Linear(23, 64) -> sinusoidal pos -> 2 Post-LN Encoder layers -> learned pool -> Head."""

    def __init__(self, seed: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        self.input_proj = nn.Linear(23, 64)
        self.pos_enc = SinusoidalPositionalEncoding(seq_len=42, d_model=64)

        # 2 post-norm encoder layers
        encoder_layer1 = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=dropout,
            activation="gelu",
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,  # Post-norm as specified
        )
        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=dropout,
            activation="gelu",
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
        )

        # Explicit initialization of encoder layers
        init_weights_xavier(self.input_proj, generator)
        for layer in [encoder_layer1, encoder_layer2]:
            init_weights_xavier(layer.self_attn.in_proj_weight, generator) if hasattr(layer.self_attn, 'in_proj_weight') else None
            init_weights_xavier(layer.linear1, generator)
            init_weights_xavier(layer.linear2, generator)
            init_weights_xavier(layer.norm1, generator)
            init_weights_xavier(layer.norm2, generator)

        self.encoder = nn.TransformerEncoder(
            encoder_layer1,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.encoder.layers[0] = encoder_layer1
        self.encoder.layers[1] = encoder_layer2

        self.pool = LearnedAttentionPooling(d_model=64, generator=generator)
        self.fc_head = nn.Linear(64, 128)
        self.ln_head = nn.LayerNorm(128, eps=1e-5)
        self.out_head = nn.Linear(128, 1)

        init_weights_xavier(self.fc_head, generator)
        init_weights_xavier(self.ln_head, generator)
        init_weights_xavier(self.out_head, generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 42, 23)
        if x.dim() != 3 or x.size(1) != 42 or x.size(2) != 23:
            raise ValueError(f"TransformerAnnual requires input shape (B, 42, 23), got {x.shape}")
        h = self.input_proj(x)  # (B, 42, 64)
        h = self.pos_enc(h)     # (B, 42, 64)
        h = self.encoder(h)     # (B, 42, 64)
        pooled = self.pool(h)   # (B, 64)
        latent = self.ln_head(self.fc_head(pooled))  # (B, 128)
        out = self.out_head(latent).squeeze(-1)       # (B,)
        return out
