from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self.score(values).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(values * weights, dim=1)


class HierarchicalLSTMCycleModel(nn.Module):
    """
    Long-context cycle model with daily and patch-level recurrence.

    Input shape follows the existing project contract: [batch, features, time].
    Output keys intentionally match CycleReasoningModel so trainer/evaluator code
    can switch backbones without changing downstream behavior.
    """

    def __init__(
        self,
        input_dim: int,
        future_target_dim: int,
        action_dim: int = 4,
        lstm_hidden_dim: int = 96,
        latent_dim: int = 128,
        lstm_layers: int = 2,
        patch_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.future_target_dim = int(future_target_dim)
        self.patch_size = max(int(patch_size), 1)
        self.supports_reconstruction = True

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, lstm_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(lstm_hidden_dim),
        )
        recurrent_dropout = dropout if lstm_layers > 1 else 0.0
        self.daily_lstm = nn.LSTM(
            input_size=lstm_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.patch_lstm = nn.LSTM(
            input_size=lstm_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.daily_pool = AttentionPool(lstm_hidden_dim)
        self.patch_pool = AttentionPool(lstm_hidden_dim)
        fused_dim = (lstm_hidden_dim * 3) + input_dim
        self.context = nn.Sequential(
            nn.Linear(fused_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(p=dropout),
        )
        self.future_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, future_target_dim),
        )
        self.reconstruction_head = nn.Linear(lstm_hidden_dim, input_dim)
        self.action_head = nn.Sequential(
            nn.Linear(latent_dim + future_target_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(latent_dim, action_dim),
        )

    def _build_patch_tokens(self, daily_states: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, hidden_dim = daily_states.shape
        patch_count = (time_steps + self.patch_size - 1) // self.patch_size
        padded_steps = patch_count * self.patch_size
        if padded_steps != time_steps:
            pad = daily_states.new_zeros(
                batch_size,
                padded_steps - time_steps,
                hidden_dim,
            )
            daily_states = torch.cat([daily_states, pad], dim=1)

        patches = daily_states.view(
            batch_size,
            patch_count,
            self.patch_size,
            hidden_dim,
        )
        counts = daily_states.new_full(
            (patch_count,),
            float(self.patch_size),
        )
        if padded_steps != time_steps:
            valid_last = time_steps - ((patch_count - 1) * self.patch_size)
            counts[-1] = float(valid_last)
        counts = counts.clamp_min(1.0).view(1, patch_count, 1)
        return patches.sum(dim=2) / counts

    def forward(
        self,
        sequence: torch.Tensor,
        return_reconstruction: bool = False,
    ) -> dict[str, torch.Tensor]:
        daily_features = sequence.transpose(1, 2)
        latest_features = daily_features[:, -1]
        x = self.input_proj(self.input_norm(daily_features))

        daily_states, _ = self.daily_lstm(x)
        daily_context = self.daily_pool(daily_states)
        latest_state = daily_states[:, -1]

        patch_tokens = self._build_patch_tokens(daily_states)
        patch_states, _ = self.patch_lstm(patch_tokens)
        patch_context = self.patch_pool(patch_states)

        latent = self.context(
            torch.cat(
                [daily_context, patch_context, latest_state, latest_features],
                dim=1,
            )
        )
        future_pred = self.future_head(latent)
        action_logits = self.action_head(torch.cat([latent, future_pred], dim=1))
        outputs = {
            "latent": latent,
            "future_pred": future_pred,
            "action_logits": action_logits,
        }
        if return_reconstruction:
            outputs["reconstruction"] = self.reconstruction_head(daily_states).transpose(
                1,
                2,
            )
        return outputs
