# src/models/timesnet_blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimesNetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        top_k: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)

        self.conv2d = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        B, T, D = x.shape

        xf = torch.fft.rfft(x, dim=1)
        freq_energy = xf.abs().mean(dim=-1)
        effective_k = min(self.top_k, freq_energy.shape[1])
        if effective_k <= 0:
            return self.dropout(self.norm(x))

        topk = torch.topk(freq_energy, effective_k, dim=1).indices

        outputs = []

        for k in range(effective_k):
            # Convert indices to float first to avoid dtype error with mean()
            period_idx = topk[:, k].float().mean().item()
            period = max(2, T // (int(period_idx) + 1))

            pad_len = (period - T % period) % period
            x_pad = F.pad(x, (0, 0, 0, pad_len))
            Tp = x_pad.shape[1]

            x_2d = x_pad.view(B, Tp // period, period, D)
            x_2d = x_2d.permute(0, 3, 1, 2)

            y = self.conv2d(x_2d)

            y = y.permute(0, 2, 3, 1).reshape(B, Tp, D)
            outputs.append(y[:, :T])

        out = sum(outputs) / len(outputs)
        return self.dropout(self.norm(out + x))
