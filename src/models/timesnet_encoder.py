import torch
import torch.nn as nn
from .timesnet_blocks import TimesNetBlock

class TimesNetEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 4,
        embed_dim: int = 64,
        num_layers: int = 2,
        top_k: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = nn.Linear(in_dim, embed_dim)
        
        self.layers = nn.ModuleList([
            TimesNetBlock(embed_dim, top_k=top_k, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        # Linear projection to embedding space
        x = self.patch_embed(x)
        
        # Pass through TimesNet layers
        for layer in self.layers:
            x = layer(x)
            
        return self.norm(x)
