from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        scores = self.score(values).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(values * weights, dim=1)


class HierarchicalPatchTransformerCycleModel(nn.Module):
    """
    Patch-based Transformer backbone that preserves the model contract used elsewhere.

    - Transformer encoder over daily tokens
    - Patch pooling into patch tokens
    - Attention pooling over daily and patch states
    - Persistent learnable memory slots read by cross-attention
    """

    def __init__(
        self,
        input_dim: int,
        future_target_dim: int,
        action_dim: int = 4,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 4,
        patch_size: int = 5,
        latent_dim: int = 128,
        num_memory_slots: int = 8,
        dropout: float = 0.1,
        memory_mode: str = "legacy_ema",
        memory_update_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.future_target_dim = int(future_target_dim)
        self.patch_size = max(int(patch_size), 1)
        self.supports_reconstruction = True

        # input projection
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, d_model)

        # positional embeddings for sequence length (max 512)
        self.pos_embed = nn.Parameter(torch.zeros(512, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # patch aggregation and small transformer for patches
        self.patch_proj = nn.Linear(d_model, d_model)
        self.patch_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=max(1, nhead // 2),
                dim_feedforward=d_model * 2,
                batch_first=True,
            ),
            num_layers=max(1, num_layers // 2),
        )

        self.daily_pool = AttentionPool(d_model)
        self.patch_pool = AttentionPool(d_model)

        # ``legacy_ema`` reproduces prior checkpoints. ``static_parameter`` is
        # the Phase 5 contract: differentiable and invariant to batch order.
        if memory_mode not in {"legacy_ema", "static_parameter"}:
            raise ValueError("memory_mode must be 'legacy_ema' or 'static_parameter'.")
        self.memory_mode = memory_mode
        self.num_memory_slots = int(num_memory_slots)
        self.memory_slots = nn.Parameter(torch.randn(self.num_memory_slots, d_model))
        # dynamic memory state saved as a buffer; starts as a copy of the parameter template
        self.register_buffer("memory_state", self.memory_slots.detach().clone())
        # memory attention for reads
        self.memory_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=max(1, nhead), batch_first=True
        )
        # project the fused query into the memory embed dimension
        self.memory_query_proj = nn.Linear((d_model * 3) + input_dim, d_model)

        # write network that maps (mem_read, latent) -> slot-wise update vector
        self.write_net = nn.Sequential(
            nn.Linear(d_model + latent_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # runtime update hyperparameter (fractional)
        self.memory_update_rate = float(memory_update_rate)
        self.enable_memory_update = memory_mode == "legacy_ema"

        fused_dim = d_model * 3 + input_dim + d_model  # extra memory read
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
        self.reconstruction_head = nn.Linear(d_model, input_dim)
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
                batch_size, padded_steps - time_steps, hidden_dim
            )
            daily_states = torch.cat([daily_states, pad], dim=1)

        patches = daily_states.view(
            batch_size, patch_count, self.patch_size, hidden_dim
        )
        counts = daily_states.new_full((patch_count,), float(self.patch_size))
        if padded_steps != time_steps:
            valid_last = time_steps - ((patch_count - 1) * self.patch_size)
            counts[-1] = float(valid_last)
        counts = counts.clamp_min(1.0).view(1, patch_count, 1)
        return patches.sum(dim=2) / counts

    def forward(
        self, sequence: torch.Tensor, return_reconstruction: bool = False
    ) -> dict[str, torch.Tensor]:
        # sequence: [B, features, time]
        daily_features = sequence.transpose(1, 2)  # [B, T, F]
        latest_features = daily_features[:, -1]
        x = self.input_proj(self.input_norm(daily_features))  # [B, T, d_model]

        T = x.size(1)
        if T > self.pos_embed.size(0):
            raise ValueError(
                "Sequence length exceeds pos_embed maximum; increase pos_embed size"
            )
        pos = self.pos_embed[:T].unsqueeze(0)
        x = x + pos

        # transformer encoder
        daily_states = self.transformer(x)  # [B, T, d_model]
        daily_context = self.daily_pool(daily_states)
        latest_state = daily_states[:, -1]

        # patch tokens
        patch_tokens = self._build_patch_tokens(daily_states)
        patch_tokens = self.patch_proj(patch_tokens)
        patch_states = self.patch_transformer(patch_tokens)
        patch_context = self.patch_pool(patch_states)

        # memory read: cross-attend from fused query to memory slots
        # build a fused query per batch and project to d_model for memory attention
        query_in = torch.cat(
            [daily_context, patch_context, latest_state, latest_features], dim=1
        )  # [B, Q]
        query = self.memory_query_proj(query_in).unsqueeze(1)  # [B,1,d_model]
        # memory slots: [S, d_model] -> expand to [B, S, d_model]
        memory_source = (
            self.memory_slots
            if self.memory_mode == "static_parameter"
            else self.memory_state
        )
        mem = memory_source.unsqueeze(0).expand(query.size(0), -1, -1)
        # MultiheadAttention expects (batch, seq, embed) when batch_first=True
        mem_read, _ = self.memory_attn(query, mem, mem)
        mem_read = mem_read.squeeze(1)

        latent = self.context(
            torch.cat(
                [daily_context, patch_context, latest_state, latest_features, mem_read],
                dim=1,
            )
        )
        future_pred = self.future_head(latent)
        action_logits = self.action_head(torch.cat([latent, future_pred], dim=1))

        # Memory write: compute a write vector per batch and aggregate into per-slot updates
        try:
            if (
                getattr(self, "enable_memory_update", False)
                and float(getattr(self, "memory_update_rate", 0.0)) > 0.0
            ):
                write_vec = self.write_net(
                    torch.cat([mem_read, latent], dim=1)
                )  # [B, d_model]
                # slot attention weights: [S, B] = [S, d] @ [d, B]
                slot_logits = torch.matmul(memory_source, write_vec.T)
                slot_weights = torch.softmax(
                    slot_logits, dim=0
                )  # normalize across slots for each sample
                # aggregate weighted writes into per-slot updates [S, d]
                slot_updates = slot_weights @ write_vec
                with torch.no_grad():
                    if torch.isfinite(slot_updates).all():
                        candidate = (
                            self.memory_state * (1.0 - float(self.memory_update_rate))
                            + float(self.memory_update_rate) * slot_updates
                        )
                        if torch.isfinite(candidate).all():
                            self.memory_state.copy_(candidate)
        except Exception:
            # be conservative: never fail the forward pass on memory update errors
            pass

        outputs = {
            "latent": latent,
            "future_pred": future_pred,
            "action_logits": action_logits,
        }
        if return_reconstruction:
            outputs["reconstruction"] = self.reconstruction_head(
                daily_states
            ).transpose(1, 2)
        return outputs

    def initialise_static_memory_from_runtime(self) -> None:
        """Copy a legacy checkpoint's runtime memory into trainable slots."""
        with torch.no_grad():
            self.memory_slots.copy_(self.memory_state)
