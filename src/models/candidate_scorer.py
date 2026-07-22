from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateScorer(nn.Module):
    """Simple MLP scorer that outputs a scalar score in [0,1]."""

    def __init__(self, input_dim: int = 13, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        out = self.net(x).squeeze(-1)
        return torch.sigmoid(out)
