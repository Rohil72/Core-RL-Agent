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


@dataclass
class LossBreakdown:
    total: torch.Tensor
    regression: torch.Tensor
    analogue: torch.Tensor
    ranking: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor
    valid_pair_count: int
    eligible_anchor_count: int
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
    
    try:
        up_idx = target_names.index("future_max_return_63")
        dn_idx = target_names.index("future_min_return_63")
        pq_idx = target_names.index("event_upside_before_drawdown_126")
    except ValueError:
        return LossBreakdown(
            total=config.lambda_reg * reg_loss,
            regression=reg_loss,
            analogue=zero,
            ranking=zero,
            variance=zero,
            covariance=zero,
            valid_pair_count=0,
            eligible_anchor_count=0,
            latent_norm_mean=0.0,
            latent_norm_std=0.0,
            effective_latent_rank=0.0,
        )

    analogue_vec = torch.stack([
        norm_targets[:, up_idx],
        norm_targets[:, dn_idx],
        norm_targets[:, pq_idx]
    ], dim=1)
    
    # weights: upside 1.0, downside 1.0, path_quality 0.50
    analogue_weights = torch.tensor([1.0, 1.0, 0.5], device=device).unsqueeze(0)
    
    batch_size = latent.size(0)
    
    # Pair validity mask
    valid_samples = torch.isfinite(analogue_vec).all(dim=1)
    
    valid_pair_count = 0
    eligible_anchor_count = 0
    analogue_loss = zero
    ranking_loss = zero
    var_loss = zero
    cov_loss = zero
    
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

    if config.lambda_analogue > 0:
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
            
            # Identify valid cross-ticker candidates for each anchor
            for i in range(batch_size):
                if not valid_samples[i]:
                    continue
                    
                is_cross_ticker = torch.tensor(
                    [ticker[i] != ticker[j] for j in range(batch_size)], 
                    device=device
                )
                valid_candidates = valid_samples & is_cross_ticker
                
                n_candidates = valid_candidates.sum().item()
                if n_candidates < 2:
                    continue
                    
                eligible_anchor_count += 1
                valid_pair_count += n_candidates
                
                cand_idx = torch.where(valid_candidates)[0]
                
                # Softmax of negative distances
                D_y_anc = D_y_full[i, cand_idx]
                D_z_anc = D_z_full[i, cand_idx]
                
                P_y = F.softmax(-D_y_anc / 0.50, dim=0)
                log_P_z = F.log_softmax(-D_z_anc / 0.20, dim=0)
                
                # KL Divergence
                kl = F.kl_div(log_P_z, P_y, reduction="sum")
                total_analogue += kl
                
            if eligible_anchor_count > 0:
                # Leave analogue loss in FP32; do not cast back to FP16
                analogue_loss = total_analogue / eligible_anchor_count

    total_loss = (
        config.lambda_reg * reg_loss +
        config.lambda_analogue * analogue_loss +
        config.lambda_rank * ranking_loss +
        config.lambda_var * (var_loss + cov_loss)
    )
    
    return LossBreakdown(
        total=total_loss,
        regression=reg_loss,
        analogue=analogue_loss,
        ranking=ranking_loss,
        variance=var_loss,
        covariance=cov_loss,
        valid_pair_count=valid_pair_count,
        eligible_anchor_count=eligible_anchor_count,
        latent_norm_mean=latent_norm_mean,
        latent_norm_std=latent_norm_std,
        effective_latent_rank=effective_latent_rank,
    )



