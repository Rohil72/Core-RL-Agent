from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                channels, channels, kernel_size=3, padding=padding, dilation=dilation
            ),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.net(x) + x))


class TemporalScaleEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        dilations: list[int],
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(ResidualTemporalBlock(hidden_dim, dilation=d) for d in dilations)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x))
        pooled = self.pool(h).squeeze(-1)
        latest = h[:, :, -1]
        return self.out_proj(torch.cat([pooled, latest], dim=1))


class CycleReasoningModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        future_target_dim: int,
        action_dim: int = 4,
        window_sizes: list[int] | None = None,
        branch_hidden_dim: int = 64,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.window_sizes = window_sizes or [21, 63, 252]
        self.branches = nn.ModuleDict(
            {
                str(window): TemporalScaleEncoder(
                    in_channels=input_dim,
                    hidden_dim=branch_hidden_dim,
                    dilations=[1, 2, 4],
                )
                for window in self.window_sizes
            }
        )
        fused_dim = branch_hidden_dim * len(self.window_sizes) + input_dim
        self.context = nn.Sequential(
            nn.Linear(fused_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(p=0.1),
        )
        self.future_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, future_target_dim),
        )
        self.action_head = nn.Sequential(
            nn.Linear(latent_dim + future_target_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(latent_dim, action_dim),
        )

    def forward(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        branch_latents = []
        for window in self.window_sizes:
            branch_latents.append(self.branches[str(window)](sequence[:, :, -window:]))
        latest = sequence[:, :, -1]
        latent = self.context(torch.cat([*branch_latents, latest], dim=1))
        future_pred = self.future_head(latent)
        action_logits = self.action_head(torch.cat([latent, future_pred], dim=1))
        return {
            "latent": latent,
            "future_pred": future_pred,
            "action_logits": action_logits,
        }
