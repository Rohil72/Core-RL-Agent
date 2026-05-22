from typing import Any

import torch.nn as nn

from src.models.cycle_reasoning_model import CycleReasoningModel
from src.models.hierarchical_lstm_model import HierarchicalLSTMCycleModel


def build_cycle_model(
    model_cfg: dict[str, Any],
    input_dim: int,
    future_target_dim: int,
    action_dim: int = 4,
) -> nn.Module:
    encoder = str(model_cfg.get("encoder", "temporal_cnn")).lower()
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

    if encoder in {"hierarchical_lstm", "lstm"}:
        hidden_dim = int(
            model_cfg.get("lstm_hidden_dim", model_cfg.get("branch_hidden_dim", 96))
        )
        return HierarchicalLSTMCycleModel(
            input_dim=input_dim,
            future_target_dim=future_target_dim,
            action_dim=action_dim,
            lstm_hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            lstm_layers=int(model_cfg.get("lstm_layers", 2)),
            patch_size=int(model_cfg.get("patch_size", 5)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )

    raise ValueError(f"Unsupported model encoder: {encoder}")


__all__ = [
    "CycleReasoningModel",
    "HierarchicalLSTMCycleModel",
    "build_cycle_model",
]
