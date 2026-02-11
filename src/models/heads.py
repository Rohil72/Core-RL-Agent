# src/models/heads.py

import torch
import torch.nn as nn


class MaskedReconstructionHead(nn.Module):
    def __init__(self, embed_dim: int, output_dim: int = 4):
        super().__init__()
        self.proj = nn.Linear(embed_dim, output_dim)

    def forward(self, z):
        return self.proj(z)


class FutureLatentHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, z):
        return self.proj(z)
