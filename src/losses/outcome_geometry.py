from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutcomeGeometryLossConfig:
    lambda_reg: float = 1.0
    lambda_analogue: float = 0.0
    lambda_rank: float = 0.0
    lambda_var: float = 0.0
    is_supcon_control: bool = False
    is_triplet_control: bool = False
    triplet_margin: float = 0.20
    future_loss_weight: float = 1.0
    max_return_loss_weight: float = 1.0
    min_return_loss_weight: float = 1.0
    analogue_target_names: tuple[str, ...] = (
        "future_max_return_63",
        "future_min_return_63",
        "event_upside_before_drawdown_126",
    )
    analogue_target_weights: tuple[float, ...] = (1.0, 1.0, 0.5)
    analogue_candidate_mode: str = "cross_ticker"
    minimum_year_gap: int = 0
    balance_candidate_domains: bool = False
    lambda_transport: float = 0.0
    transport_positive_quantile: float = 0.25
    transport_negative_quantile: float = 0.75
    transport_margin: float = 0.25
    variance_target: float = 0.20
    enable_variance_regularizer: bool = False


@dataclass
class LossBreakdown:
    total: torch.Tensor
    regression: torch.Tensor
    analogue: torch.Tensor
    ranking: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor
    transport: torch.Tensor
    valid_pair_count: int
    eligible_anchor_count: int
    transport_triplet_count: int
    latent_norm_mean: float
    latent_norm_std: float
    effective_latent_rank: float


class OutcomeTargetNormalizer:
    def __init__(self) -> None:
        self.medians: dict[str, float] = {}
        self.iqrs: dict[str, float] = {}
        self.is_fitted = False

    def fit(self, targets: torch.Tensor, target_names: Sequence[str]) -> None:
        """Fit strictly on finite training targets."""
        for i, name in enumerate(target_names):
            col = targets[:, i]
            finite_mask = torch.isfinite(col)
            if not finite_mask.any():
                self.medians[name] = 0.0
                self.iqrs[name] = 1.0
                continue
            finite_vals = col[finite_mask]
            
            # Compute median and IQR
            q25, q50, q75 = torch.quantile(finite_vals.float(), torch.tensor([0.25, 0.50, 0.75], device=finite_vals.device))
            iqr = max(float(q75 - q25), 1e-6)
            
            self.medians[name] = float(q50)
            self.iqrs[name] = iqr
            
        self.is_fitted = True

    def normalize(self, targets: torch.Tensor, target_names: Sequence[str]) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("OutcomeTargetNormalizer must be fitted on training data before use.")
        out = torch.zeros_like(targets)
        for i, name in enumerate(target_names):
            med = self.medians.get(name, 0.0)
            iqr = self.iqrs.get(name, 1.0)
            out[:, i] = (targets[:, i] - med) / iqr
        return out


def compute_market_memory_loss(
    latent: torch.Tensor,
    future_prediction: torch.Tensor,
    future_target: torch.Tensor,
    ticker: Sequence[str],
    target_names: Sequence[str],
    target_normalizer: OutcomeTargetNormalizer,
    config: OutcomeGeometryLossConfig,
    market: Sequence[str] | None = None,
    timestamp: Sequence[str] | None = None,
) -> LossBreakdown:
    device = latent.device
    zero = torch.tensor(0.0, device=device)
    
    # Prior weighted masked-Huber calculation
    reg_loss = zero
    if config.lambda_reg > 0:
        weights = [1.0] * (future_prediction.size(1) if future_prediction.dim() > 1 else 1)
        try:
            idx = target_names.index("future_max_return_63")
            weights[idx] = config.max_return_loss_weight
        except ValueError:
            pass
        try:
            idx = target_names.index("future_min_return_63")
            weights[idx] = config.min_return_loss_weight
        except ValueError:
            pass

        future_loss_total = future_prediction.new_tensor(0.0)
        total_w = 0.0
        for k, w in enumerate(weights):
            val_mask = torch.isfinite(future_target[:, k])
            if val_mask.any():
                loss_k = F.smooth_l1_loss(
                    future_prediction[val_mask, k], future_target[val_mask, k], reduction="mean"
                )
                future_loss_total = future_loss_total + float(w) * loss_k
                total_w += float(w)
        
        if total_w > 0:
            reg_loss = (future_loss_total / total_w) * config.future_loss_weight

    norm_targets = target_normalizer.normalize(future_target, target_names)
    
    if len(config.analogue_target_names) != len(config.analogue_target_weights):
        raise ValueError("analogue target names and weights must have equal length.")
    missing_analogue_targets = [
        name for name in config.analogue_target_names if name not in target_names
    ]
    if missing_analogue_targets and (
        config.lambda_analogue > 0 or config.lambda_transport > 0
    ):
        raise ValueError(
            "Enabled outcome geometry is missing targets: "
            + ", ".join(missing_analogue_targets)
        )
    if missing_analogue_targets:
        return LossBreakdown(
            total=config.lambda_reg * reg_loss,
            regression=reg_loss,
            analogue=zero,
            ranking=zero,
            variance=zero,
            covariance=zero,
            transport=zero,
            valid_pair_count=0,
            eligible_anchor_count=0,
            transport_triplet_count=0,
            latent_norm_mean=0.0,
            latent_norm_std=0.0,
            effective_latent_rank=0.0,
        )
    analogue_indices = [target_names.index(name) for name in config.analogue_target_names]

    analogue_vec = torch.stack(
        [norm_targets[:, idx] for idx in analogue_indices], dim=1
    )
    analogue_weights = torch.tensor(
        config.analogue_target_weights, device=device
    ).unsqueeze(0)
    
    batch_size = latent.size(0)
    
    # Pair validity mask
    valid_samples = torch.isfinite(analogue_vec).all(dim=1)
    
    valid_pair_count = 0
    eligible_anchor_count = 0
    analogue_loss = zero
    ranking_loss = zero
    var_loss = zero
    cov_loss = zero
    transport_loss = zero
    transport_triplet_count = 0
    
    latent_norms = latent.norm(p=2, dim=1)
    latent_norm_mean = float(latent_norms.mean().item()) if latent_norms.numel() > 0 else 0.0
    latent_norm_std = float(latent_norms.std().item()) if latent_norms.numel() > 1 else 0.0
    
    # Compute effective rank of latents
    effective_latent_rank = 0.0
    if batch_size > 1:
        try:
            _, S, _ = torch.linalg.svd(latent.detach(), full_matrices=False)
            norm_S = S / S.sum()
            entropy = -(norm_S * torch.log(norm_S + 1e-9)).sum()
            effective_latent_rank = float(torch.exp(entropy).item())
        except Exception:
            pass

    supported_modes = {
        "cross_ticker",
        "cross_period",
        "cross_market_or_period",
        "cross_market_and_period",
    }
    if config.analogue_candidate_mode not in supported_modes:
        raise ValueError(
            f"Unsupported analogue_candidate_mode {config.analogue_candidate_mode!r}."
        )
    if config.minimum_year_gap < 0:
        raise ValueError("minimum_year_gap cannot be negative.")
    if not (
        0.0 <= config.transport_positive_quantile
        < config.transport_negative_quantile
        <= 1.0
    ):
        raise ValueError(
            "Transport quantiles must satisfy 0 <= positive < negative <= 1."
        )
    if config.transport_margin < 0:
        raise ValueError("transport_margin cannot be negative.")
    if config.variance_target <= 0:
        raise ValueError("variance_target must be positive.")
    if config.lambda_transport > 0 and timestamp is None:
        raise ValueError("Temporal transport loss requires batch timestamps.")

    years = None
    parsed_years: list[int] | None = None
    if timestamp is not None:
        try:
            parsed_years = [int(str(value)[:4]) for value in timestamp]
            years = torch.tensor(parsed_years, device=device)
        except (TypeError, ValueError) as exc:
            raise ValueError("Batch timestamps must begin with a four-digit year.") from exc

    ticker_values = [str(value) for value in ticker]
    cross_ticker_matrix = torch.tensor(
        [
            [ticker_values[i] != ticker_values[j] for j in range(batch_size)]
            for i in range(batch_size)
        ],
        device=device,
    )
    market_values = [str(value) for value in market] if market is not None else None
    cross_market_matrix = (
        torch.tensor(
            [
                [market_values[i] != market_values[j] for j in range(batch_size)]
                for i in range(batch_size)
            ],
            device=device,
        )
        if market_values is not None
        else torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
    )
    cross_period_matrix = (
        torch.abs(years.unsqueeze(1) - years.unsqueeze(0))
        >= max(config.minimum_year_gap, 1)
        if years is not None
        else None
    )

    def candidate_mask(anchor: int, *, transport: bool = False) -> torch.Tensor:
        eligible = valid_samples & cross_ticker_matrix[anchor]
        mode = config.analogue_candidate_mode
        if mode == "cross_ticker" and not transport:
            return eligible
        if cross_period_matrix is None:
            raise ValueError(f"{mode} analogue candidates require batch timestamps.")
        cross_period = cross_period_matrix[anchor]
        cross_market = cross_market_matrix[anchor]
        if mode == "cross_market_and_period":
            return eligible & cross_period & cross_market
        if mode == "cross_market_or_period":
            return eligible & (cross_period | cross_market)
        return eligible & cross_period

    if config.lambda_analogue > 0 or config.lambda_transport > 0:
        with torch.amp.autocast(device_type=device.type, enabled=False):
            # Compute distance matrix in float32
            analogue_vec_f32 = analogue_vec.to(torch.float32)
            latent_f32 = latent.to(torch.float32)
            analogue_weights_f32 = analogue_weights.to(torch.float32)
            
            diff_y = analogue_vec_f32.unsqueeze(1) - analogue_vec_f32.unsqueeze(0)
            D_y_full = (diff_y ** 2 * analogue_weights_f32).sum(dim=-1)
            
            # Smooth normalization denominator guard: sqrt(sum(x^2) + 1e-8)
            z_norm = latent_f32 / torch.sqrt(torch.sum(latent_f32**2, dim=-1, keepdim=True) + 1e-8)
            diff_z = z_norm.unsqueeze(1) - z_norm.unsqueeze(0)
            D_z_full = (diff_z ** 2).sum(dim=-1)
            
            total_analogue = torch.tensor(0.0, device=device, dtype=torch.float32)
            total_transport = torch.tensor(0.0, device=device, dtype=torch.float32)
            
            # Identify valid cross-ticker candidates for each anchor
            for i in range(batch_size):
                if not valid_samples[i]:
                    continue
                    
                valid_candidates = candidate_mask(i)
                
                n_candidates = valid_candidates.sum().item()
                if n_candidates < 2:
                    continue
                    
                cand_idx = torch.where(valid_candidates)[0]
                D_y_anc = D_y_full[i, cand_idx]
                D_z_anc = D_z_full[i, cand_idx]
                if config.lambda_analogue > 0:
                    eligible_anchor_count += 1
                    valid_pair_count += n_candidates
                    outcome_logits = -D_y_anc / 0.50
                    if config.balance_candidate_domains:
                        domains = [
                            f"{market_values[j] if market_values is not None else 'all'}|"
                            f"{parsed_years[j] if parsed_years is not None else 'all'}"
                            for j in cand_idx.tolist()
                        ]
                        counts = {domain: domains.count(domain) for domain in set(domains)}
                        correction = torch.tensor(
                            [float(counts[domain]) for domain in domains],
                            device=device,
                            dtype=torch.float32,
                        ).log()
                        outcome_logits = outcome_logits - correction
                    P_y = F.softmax(outcome_logits, dim=0)
                    log_P_z = F.log_softmax(-D_z_anc / 0.20, dim=0)
                    total_analogue += F.kl_div(log_P_z, P_y, reduction="sum")

                if config.lambda_transport > 0:
                    transport_candidates = torch.where(candidate_mask(i, transport=True))[0]
                    base_candidates = torch.where(
                        valid_samples & cross_ticker_matrix[i]
                    )[0]
                    if len(transport_candidates) == 0 or len(base_candidates) < 2:
                        continue
                    candidate_outcomes = D_y_full[i, base_candidates]
                    positive_cutoff = torch.quantile(
                        candidate_outcomes, config.transport_positive_quantile
                    )
                    negative_cutoff = torch.quantile(
                        candidate_outcomes, config.transport_negative_quantile
                    )
                    positive_idx = transport_candidates[
                        D_y_full[i, transport_candidates] <= positive_cutoff
                    ]
                    negative_idx = base_candidates[
                        D_y_full[i, base_candidates] >= negative_cutoff
                    ]
                    if len(positive_idx) == 0 or len(negative_idx) == 0:
                        continue
                    positive = positive_idx[
                        torch.argmin(D_y_full[i, positive_idx])
                    ]
                    negative = negative_idx[
                        torch.argmin(D_z_full[i, negative_idx])
                    ]
                    total_transport += F.relu(
                        D_z_full[i, positive]
                        - D_z_full[i, negative]
                        + config.transport_margin
                    )
                    transport_triplet_count += 1
                
            if eligible_anchor_count > 0:
                analogue_loss = total_analogue / eligible_anchor_count
            if transport_triplet_count > 0:
                transport_loss = total_transport / transport_triplet_count

    if config.enable_variance_regularizer and config.lambda_var > 0 and batch_size > 1:
        latent_f32 = latent.to(torch.float32)
        centred = latent_f32 - latent_f32.mean(dim=0, keepdim=True)
        std = torch.sqrt(centred.var(dim=0, unbiased=False) + 1e-4)
        var_loss = F.relu(config.variance_target - std).mean()
        covariance = centred.T @ centred / max(batch_size - 1, 1)
        off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
        cov_loss = off_diagonal.square().sum() / max(latent_f32.size(1), 1)

    total_loss = (
        config.lambda_reg * reg_loss +
        config.lambda_analogue * analogue_loss +
        config.lambda_rank * ranking_loss +
        config.lambda_var * (var_loss + cov_loss) +
        config.lambda_transport * transport_loss
    )
    
    return LossBreakdown(
        total=total_loss,
        regression=reg_loss,
        analogue=analogue_loss,
        ranking=ranking_loss,
        variance=var_loss,
        covariance=cov_loss,
        transport=transport_loss,
        valid_pair_count=valid_pair_count,
        eligible_anchor_count=eligible_anchor_count,
        transport_triplet_count=transport_triplet_count,
        latent_norm_mean=latent_norm_mean,
        latent_norm_std=latent_norm_std,
        effective_latent_rank=effective_latent_rank,
    )
