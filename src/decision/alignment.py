from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from src.decision.adapter import DecisionAdapter
from src.decision.losses import DecisionLossConfig, decision_alignment_loss


@dataclass(frozen=True)
class AlignmentConfig:
    epochs: int = 4
    encoder_learning_rate: float = 3e-5
    adapter_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    maximum_future_mae_regression: float = 0.05
    encoder_parameter_scope: str = "approved_blocks"

    def __post_init__(self) -> None:
        if self.encoder_parameter_scope not in {"approved_blocks", "memory_only"}:
            raise ValueError("encoder_parameter_scope must be 'approved_blocks' or 'memory_only'.")


class AlignedDecisionModel(nn.Module):
    """Compose the market-state encoder and decision adapter without merging contracts."""

    def __init__(self, encoder: nn.Module, adapter: DecisionAdapter) -> None:
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter

    def forward(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.encoder(sequence)
        state.update(self.adapter(state["latent"]))
        return state


def configure_alignment_parameters(
    model: AlignedDecisionModel,
    scope: str = "approved_blocks",
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Freeze the encoder and expose only the configured alignment parameter scope."""
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    if scope == "memory_only":
        prefixes = ("memory_slots", "memory_attn", "memory_query_proj")
    elif scope == "approved_blocks":
        prefixes = (
            "transformer.layers.3",
            "patch_transformer.layers.1",
            "memory_slots",
            "memory_attn",
            "memory_query_proj",
            "context",
            "future_head",
        )
    else:
        raise ValueError(f"Unknown alignment parameter scope: {scope}")
    for name, parameter in model.encoder.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(True)
    for parameter in model.adapter.parameters():
        parameter.requires_grad_(True)
    encoder_parameters = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    adapter_parameters = [parameter for parameter in model.adapter.parameters() if parameter.requires_grad]
    if not encoder_parameters or not adapter_parameters:
        raise RuntimeError("Alignment parameter selection produced an empty optimizer group.")
    return encoder_parameters, adapter_parameters


def _project_conflicting(first: list[torch.Tensor], second: list[torch.Tensor]) -> list[torch.Tensor]:
    dot = sum((a * b).sum() for a, b in zip(first, second))
    norm = sum((b * b).sum() for b in second).clamp_min(1e-12)
    if dot >= 0:
        return first
    scale = dot / norm
    return [a - scale * b for a, b in zip(first, second)]


def pcgrad_backward(losses: Iterable[torch.Tensor], parameters: Iterable[nn.Parameter]) -> None:
    """Apply two-or-more objective PCGrad while retaining parameters absent from a task."""
    params = list(parameters)
    loss_list = list(losses)
    task_gradients: list[list[torch.Tensor]] = []
    for index, loss in enumerate(loss_list):
        gradients = torch.autograd.grad(
            loss, params, retain_graph=index < len(loss_list) - 1, allow_unused=True
        )
        task_gradients.append([
            gradient if gradient is not None else torch.zeros_like(parameter)
            for gradient, parameter in zip(gradients, params)
        ])
    projected: list[list[torch.Tensor]] = []
    for i, gradients in enumerate(task_gradients):
        current = gradients
        for j, other in enumerate(task_gradients):
            if i != j:
                current = _project_conflicting(current, other)
        projected.append(current)
    for parameter_index, parameter in enumerate(params):
        parameter.grad = torch.stack([task[parameter_index] for task in projected]).mean(dim=0)


def target_indices(target_names: list[str]) -> dict[str, int]:
    required = (
        "future_return_21", "future_return_63", "future_return_126",
        "future_max_return_63", "future_min_return_63", "event_upside_before_drawdown_126",
    )
    missing = [name for name in required if name not in target_names]
    if missing:
        raise ValueError(f"Alignment requires future targets: {missing}")
    return {name: target_names.index(name) for name in required}


def alignment_objectives(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    timestamps: list[str],
    tickers: list[str],
    indices: dict[str, int],
    loss_config: DecisionLossConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Build future-preservation and economic-neighbour objectives for one batch."""
    finite = torch.isfinite(targets)
    future_error = F.smooth_l1_loss(outputs["future_pred"][finite], targets[finite])
    utility = (
        targets[:, indices["future_return_63"]]
        - 0.50 * targets[:, indices["future_min_return_63"]].abs()
        - 0.0063
        - 0.0020
    )
    outcomes = torch.stack(
        [
            targets[:, indices["future_return_21"]],
            targets[:, indices["future_return_63"]],
            targets[:, indices["future_return_126"]],
            targets[:, indices["future_max_return_63"]],
            targets[:, indices["future_min_return_63"]],
            targets[:, indices["event_upside_before_drawdown_126"]],
        ],
        dim=1,
    )
    date_ids = torch.as_tensor(_factorize(timestamps), device=targets.device)
    ticker_ids = torch.as_tensor(_factorize(tickers), device=targets.device)
    decision, parts = decision_alignment_loss(
        outputs, utility, outcomes, date_ids, ticker_ids,
        outputs.get("quantiles", (0.10, 0.50, 0.90)), loss_config,
    )
    return future_error, decision, parts


def _factorize(values: list[str]) -> list[int]:
    mapping: dict[str, int] = {}
    return [mapping.setdefault(str(value), len(mapping)) for value in values]


def save_aligned_checkpoint(path: str | Path, model: AlignedDecisionModel, metadata: dict[str, object]) -> None:
    """Persist encoder and adapter independently so either can be frozen later."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state": model.encoder.state_dict(),
            "adapter": model.adapter.checkpoint(),
            "metadata": metadata,
        },
        destination,
    )
