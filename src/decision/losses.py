from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DecisionLossConfig:
    quantile_weight: float = 1.0
    ranking_weight: float = 0.50
    analogue_weight: float = 0.20
    vicreg_weight: float = 0.01
    rank_temperature: float = 0.20
    outcome_temperature: float = 0.50
    latent_temperature: float = 0.20
    vicreg_variance_target: float = 0.20


def quantile_loss(prediction: torch.Tensor, target: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
    """Pinball loss for utility-distribution estimation."""
    error = target.unsqueeze(1) - prediction
    q = prediction.new_tensor(quantiles).unsqueeze(0)
    return torch.maximum(q * error, (q - 1.0) * error).mean()


def same_date_listwise_loss(scores: torch.Tensor, utility: torch.Tensor, date_ids: torch.Tensor) -> torch.Tensor:
    """Pairwise logistic ranking restricted to alternatives available on one date."""
    same_date = date_ids[:, None] == date_ids[None, :]
    target_order = utility[:, None] > utility[None, :]
    valid = same_date & target_order & torch.isfinite(utility[:, None]) & torch.isfinite(utility[None, :])
    if not valid.any():
        return scores.sum() * 0.0
    margins = scores[:, None] - scores[None, :]
    return F.softplus(-margins[valid]).mean()


def analogue_geometry_loss(
    decision: torch.Tensor,
    outcomes: torch.Tensor,
    ticker_ids: torch.Tensor,
    outcome_temperature: float = 0.50,
    latent_temperature: float = 0.20,
) -> torch.Tensor:
    """Align cross-security neighbour distributions with multi-horizon economics."""
    valid_rows = torch.isfinite(outcomes).all(dim=1)
    cross_ticker = ticker_ids[:, None] != ticker_ids[None, :]
    pair_mask = valid_rows[:, None] & valid_rows[None, :] & cross_ticker
    losses: list[torch.Tensor] = []
    d_z = torch.cdist(decision.float(), decision.float()).square()
    d_y = torch.cdist(outcomes.float(), outcomes.float()).square()
    for anchor in range(len(decision)):
        candidates = pair_mask[anchor]
        if candidates.sum() < 2:
            continue
        target_prob = F.softmax(-d_y[anchor, candidates] / outcome_temperature, dim=0)
        latent_log_prob = F.log_softmax(-d_z[anchor, candidates] / latent_temperature, dim=0)
        losses.append(F.kl_div(latent_log_prob, target_prob, reduction="sum"))
    return torch.stack(losses).mean() if losses else decision.sum() * 0.0


def vicreg_loss(decision: torch.Tensor, variance_target: float = 0.20) -> torch.Tensor:
    """Prevent dimensional collapse without imposing arbitrary class labels."""
    centered = decision - decision.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(variance_target - std).mean()
    covariance = centered.T @ centered / max(len(decision) - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    return variance + off_diagonal.square().sum() / decision.size(1)


def decision_alignment_loss(
    outputs: dict[str, torch.Tensor],
    utility: torch.Tensor,
    outcomes: torch.Tensor,
    date_ids: torch.Tensor,
    ticker_ids: torch.Tensor,
    quantiles: tuple[float, ...],
    config: DecisionLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the complete, non-placeholder Phase 5B objective."""
    q = quantile_loss(outputs["utility_quantiles"], utility, quantiles)
    rank = same_date_listwise_loss(outputs["utility_quantiles"][:, 1], utility, date_ids)
    analogue = analogue_geometry_loss(
        outputs["decision"], outcomes, ticker_ids, config.outcome_temperature, config.latent_temperature
    )
    regularity = vicreg_loss(outputs["decision"], config.vicreg_variance_target)
    total = (
        config.quantile_weight * q
        + config.ranking_weight * rank
        + config.analogue_weight * analogue
        + config.vicreg_weight * regularity
    )
    return total, {"quantile": q, "ranking": rank, "analogue": analogue, "vicreg": regularity}
