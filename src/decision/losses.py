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
    temporal_event_weight: float = 0.0
    temporal_ranking_weight: float = 0.0
    temporal_coherence_weight: float = 0.0
    temporal_rank_temperature: float = 0.10
    coherence_state_temperature: float = 0.20
    temporal_class_weighting: str = "none"
    temporal_class_weight_max: float = 4.0
    opportunity_weight: float = 0.0
    coverage_weight: float = 0.0
    opportunity_temperature: float = 0.02
    opportunity_top_k: int = 3
    minimum_positive_utility: float = 0.0
    environment_weight: float = 0.0
    environment_trim_fraction: float = 0.0
    environment_minimum_rows: int = 8

    def __post_init__(self) -> None:
        if not 0.0 <= self.environment_trim_fraction < 0.5:
            raise ValueError("environment_trim_fraction must be in [0, 0.5).")
        if self.opportunity_top_k <= 0:
            raise ValueError("opportunity_top_k must be positive.")
        if self.opportunity_temperature <= 0 or self.temporal_rank_temperature <= 0:
            raise ValueError("Loss temperatures must be positive.")
        if self.temporal_class_weighting not in {"none", "inverse_sqrt"}:
            raise ValueError("temporal_class_weighting must be 'none' or 'inverse_sqrt'.")
        if self.temporal_class_weight_max < 1.0:
            raise ValueError("temporal_class_weight_max must be at least 1.0.")


def quantile_loss(prediction: torch.Tensor, target: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
    """Pinball loss for utility-distribution estimation."""
    error = target.unsqueeze(1) - prediction
    q = prediction.new_tensor(quantiles).unsqueeze(0)
    return torch.maximum(q * error, (q - 1.0) * error).mean()


def _quantile_loss_per_row(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantiles: tuple[float, ...],
) -> torch.Tensor:
    error = target.unsqueeze(1) - prediction
    q = prediction.new_tensor(quantiles).unsqueeze(0)
    return torch.maximum(q * error, (q - 1.0) * error).mean(dim=1)


def temporal_event_probabilities(
    event_logits: torch.Tensor,
    horizon_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return event-class probabilities and monotonic cumulative event risks."""

    expected_classes = 1 + 2 * horizon_count
    if event_logits.ndim != 2 or event_logits.size(1) != expected_classes:
        raise ValueError(f"Expected event_logits with {expected_classes} classes.")
    probabilities = F.softmax(event_logits, dim=1)
    upside = probabilities[:, 1 : 1 + horizon_count].cumsum(dim=1)
    downside = probabilities[:, 1 + horizon_count :].cumsum(dim=1)
    return probabilities, torch.stack([upside, downside], dim=-1)


def temporal_class_weights(
    targets: torch.Tensor,
    class_count: int,
    mode: str = "none",
    maximum_weight: float = 4.0,
) -> torch.Tensor | None:
    """Derive capped event-class weights from the training split only."""

    if mode == "none":
        return None
    if mode != "inverse_sqrt":
        raise ValueError(f"Unsupported temporal class weighting mode: {mode}")
    counts = torch.bincount(targets.long().cpu(), minlength=class_count).float()
    if counts.sum() <= 0:
        raise ValueError("Cannot derive temporal class weights from an empty target.")
    smoothed = counts.clamp_min(1.0)
    weights = torch.sqrt(smoothed.sum() / smoothed)
    weights /= weights.mean()
    return weights.clamp(max=float(maximum_weight))


def temporal_concordance_loss(
    cumulative_risk: torch.Tensor,
    event_target: torch.Tensor,
    horizon_count: int,
    temperature: float,
) -> torch.Tensor:
    """Rank earlier observed competing events above later-at-risk examples."""

    target = event_target.long()
    event_type = torch.full_like(target, -1)
    event_bucket = torch.full_like(target, horizon_count)
    upside = (target >= 1) & (target <= horizon_count)
    downside = target > horizon_count
    event_type[upside] = 0
    event_type[downside] = 1
    event_bucket[upside] = target[upside] - 1
    event_bucket[downside] = target[downside] - 1 - horizon_count
    losses: list[torch.Tensor] = []
    for index in torch.where(event_type >= 0)[0].tolist():
        bucket = int(event_bucket[index])
        kind = int(event_type[index])
        later = event_bucket > bucket
        if not later.any():
            continue
        anchor = cumulative_risk[index, bucket, kind]
        comparison = cumulative_risk[later, bucket, kind]
        losses.append(F.softplus(-(anchor - comparison) / temperature).mean())
    return torch.stack(losses).mean() if losses else cumulative_risk.sum() * 0.0


def temporal_coherence_loss(
    cumulative_risk: torch.Tensor,
    decision: torch.Tensor,
    date_ids: torch.Tensor,
    ticker_ids: torch.Tensor,
    state_temperature: float,
) -> torch.Tensor:
    """Discourage unexplained one-session changes while permitting state shifts."""

    same_ticker = ticker_ids[:, None] == ticker_ids[None, :]
    adjacent_date = (date_ids[:, None] - date_ids[None, :]).abs() == 1
    pairs = torch.triu(same_ticker & adjacent_date, diagonal=1)
    if not pairs.any():
        return cumulative_risk.sum() * 0.0
    left, right = torch.where(pairs)
    state_distance = torch.linalg.vector_norm(decision[left] - decision[right], dim=1)
    gate = torch.exp(-state_distance / state_temperature).detach()
    probability_shift = (cumulative_risk[left] - cumulative_risk[right]).square().mean(dim=(1, 2))
    return (gate * probability_shift).sum() / gate.sum().clamp_min(1e-6)


def opportunity_allocation_losses(
    scores: torch.Tensor,
    utility: torch.Tensor,
    date_ids: torch.Tensor,
    cash_logit: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
    minimum_positive_utility: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train date-level stock-versus-cash allocation and expose realized regret."""

    allocation_losses: list[torch.Tensor] = []
    coverage_losses: list[torch.Tensor] = []
    regrets: list[torch.Tensor] = []
    for date_id in torch.unique(date_ids):
        members = torch.where(date_ids == date_id)[0]
        local_utility = utility[members]
        finite = torch.isfinite(local_utility)
        members, local_utility = members[finite], local_utility[finite]
        if not len(members):
            continue
        local_scores = scores[members]
        logits = torch.cat([cash_logit.reshape(1), local_scores]) / temperature
        predicted = F.softmax(logits, dim=0)
        positive = local_utility > minimum_positive_utility
        positive_count = int(positive.sum())
        target_exposure = min(1.0, positive_count / top_k)
        target = torch.zeros_like(predicted)
        target[0] = 1.0 - target_exposure
        if positive_count:
            candidate_indices = torch.where(positive)[0]
            take = min(top_k, positive_count)
            selected = candidate_indices[torch.topk(local_utility[candidate_indices], take).indices]
            selected_weights = F.softmax(local_utility[selected] / temperature, dim=0)
            target[selected + 1] = target_exposure * selected_weights
        allocation_losses.append(-(target * F.log_softmax(logits, dim=0)).sum())
        predicted_exposure = 1.0 - predicted[0]
        coverage_losses.append((predicted_exposure - target_exposure) ** 2)
        target_utility = (target[1:] * local_utility).sum()
        predicted_utility = (predicted[1:] * local_utility).sum()
        regrets.append((target_utility - predicted_utility).clamp_min(0.0))
    zero = scores.sum() * 0.0
    if not allocation_losses:
        return zero, zero, zero
    return (
        torch.stack(allocation_losses).mean(),
        torch.stack(coverage_losses).mean(),
        torch.stack(regrets).mean(),
    )


def environment_risk_variance(
    per_row_loss: torch.Tensor,
    environment_ids: torch.Tensor,
    *,
    trim_fraction: float,
    minimum_rows: int,
) -> torch.Tensor:
    """Penalize risk variation after trimming extreme within-environment losses."""

    risks: list[torch.Tensor] = []
    for environment_id in torch.unique(environment_ids):
        values = per_row_loss[environment_ids == environment_id]
        if len(values) < minimum_rows:
            continue
        keep = max(minimum_rows, int(len(values) * (1.0 - trim_fraction)))
        if keep < len(values):
            values = torch.sort(values).values[:keep]
        risks.append(values.mean())
    if len(risks) < 2:
        return per_row_loss.sum() * 0.0
    return torch.stack(risks).var(unbiased=False)


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
    temporal_target: torch.Tensor | None = None,
    temporal_weights: torch.Tensor | None = None,
    environment_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the complete, non-placeholder Phase 5B objective."""
    q = quantile_loss(outputs["utility_quantiles"], utility, quantiles)
    rank = same_date_listwise_loss(outputs["utility_quantiles"][:, 1], utility, date_ids)
    analogue = analogue_geometry_loss(
        outputs["decision"], outcomes, ticker_ids, config.outcome_temperature, config.latent_temperature
    )
    regularity = vicreg_loss(outputs["decision"], config.vicreg_variance_target)
    parts: dict[str, torch.Tensor] = {
        "quantile": q,
        "ranking": rank,
        "analogue": analogue,
        "vicreg": regularity,
    }
    total = (
        config.quantile_weight * q
        + config.ranking_weight * rank
        + config.analogue_weight * analogue
        + config.vicreg_weight * regularity
    )
    per_row_risk = _quantile_loss_per_row(outputs["utility_quantiles"], utility, quantiles)
    if config.temporal_event_weight > 0 or config.temporal_ranking_weight > 0 or config.temporal_coherence_weight > 0:
        if temporal_target is None or "event_logits" not in outputs:
            raise ValueError("Temporal losses require temporal_target and event_logits.")
        horizon_count = (outputs["event_logits"].size(1) - 1) // 2
        event_per_row = F.cross_entropy(
            outputs["event_logits"],
            temporal_target.long(),
            weight=temporal_weights,
            reduction="none",
        )
        event_loss = event_per_row.mean()
        _, cumulative_risk = temporal_event_probabilities(outputs["event_logits"], horizon_count)
        temporal_rank = temporal_concordance_loss(
            cumulative_risk,
            temporal_target,
            horizon_count,
            config.temporal_rank_temperature,
        )
        coherence = temporal_coherence_loss(
            cumulative_risk,
            outputs["decision"],
            date_ids,
            ticker_ids,
            config.coherence_state_temperature,
        )
        total = (
            total
            + config.temporal_event_weight * event_loss
            + config.temporal_ranking_weight * temporal_rank
            + config.temporal_coherence_weight * coherence
        )
        per_row_risk = per_row_risk + config.temporal_event_weight * event_per_row
        parts.update({"temporal_event": event_loss, "temporal_ranking": temporal_rank, "temporal_coherence": coherence})
    if config.opportunity_weight > 0 or config.coverage_weight > 0:
        if "cash_logit" not in outputs:
            raise ValueError("Opportunity losses require include_cash_logit=true in the adapter.")
        allocation, coverage, regret = opportunity_allocation_losses(
            outputs["utility_quantiles"][:, 1],
            utility,
            date_ids,
            outputs["cash_logit"],
            top_k=config.opportunity_top_k,
            temperature=config.opportunity_temperature,
            minimum_positive_utility=config.minimum_positive_utility,
        )
        total = total + config.opportunity_weight * allocation + config.coverage_weight * coverage
        parts.update({"opportunity": allocation, "coverage": coverage, "decision_regret": regret})
    if config.environment_weight > 0:
        if environment_ids is None:
            raise ValueError("environment_weight requires environment_ids.")
        environment = environment_risk_variance(
            per_row_risk,
            environment_ids,
            trim_fraction=config.environment_trim_fraction,
            minimum_rows=config.environment_minimum_rows,
        )
        total = total + config.environment_weight * environment
        parts["environment"] = environment
    return total, parts
