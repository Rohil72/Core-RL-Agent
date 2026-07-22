from typing import Any

import torch.nn as nn

from src.models.cycle_reasoning_model import CycleReasoningModel
from src.models.patch_transformer_model import HierarchicalPatchTransformerCycleModel


def build_cycle_model(
    model_cfg: dict[str, Any],
    input_dim: int,
    future_target_dim: int,
    action_dim: int = 4,
) -> nn.Module:
    encoder = str(model_cfg.get("encoder", "patch_transformer")).lower()
    latent_dim = int(model_cfg["latent_dim"])

    if encoder in {"temporal_cnn", "cnn", "conv1d"}:
        return CycleReasoningModel(
            input_dim=input_dim,
            future_target_dim=future_target_dim,
            action_dim=action_dim,
            window_sizes=list(model_cfg["window_sizes"]),
            branch_hidden_dim=int(model_cfg["branch_hidden_dim"]),
            latent_dim=latent_dim,
        )

    if encoder in {"patch_transformer", "transformer", "patch_tst"}:
        return HierarchicalPatchTransformerCycleModel(
            input_dim=input_dim,
            future_target_dim=future_target_dim,
            action_dim=action_dim,
            d_model=int(model_cfg.get("d_model", 96)),
            nhead=int(model_cfg.get("nhead", 4)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            patch_size=int(model_cfg.get("patch_size", 5)),
            latent_dim=latent_dim,
            num_memory_slots=int(model_cfg.get("num_memory_slots", 8)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            memory_mode=str(model_cfg.get("memory_mode", "legacy_ema")),
            memory_update_rate=float(model_cfg.get("memory_update_rate", 0.05)),
        )

    if encoder in {"hierarchical_lstm", "lstm"}:
        raise ValueError(
            "The hierarchical_lstm encoder has been removed. Use 'patch_transformer' instead."
        )

    raise ValueError(f"Unsupported model encoder: {encoder}")


__all__ = [
    "CycleReasoningModel",
    "HierarchicalPatchTransformerCycleModel",
    "build_cycle_model",
]
