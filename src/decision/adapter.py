from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DecisionAdapterConfig:
    """Architecture for the frozen-state to decision-space projection."""

    input_dim: int = 128
    hidden_dim: int = 64
    decision_dim: int = 32
    dropout: float = 0.10
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90)
    temporal_horizons: tuple[int, ...] = ()
    enforce_non_crossing_quantiles: bool = False
    include_cash_logit: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantiles", tuple(self.quantiles))
        object.__setattr__(self, "temporal_horizons", tuple(self.temporal_horizons))
        if self.enforce_non_crossing_quantiles and len(self.quantiles) != 3:
            raise ValueError("Non-crossing parameterization currently requires exactly three quantiles.")
        if tuple(sorted(self.temporal_horizons)) != self.temporal_horizons:
            raise ValueError("temporal_horizons must be strictly ordered.")
        if len(set(self.temporal_horizons)) != len(self.temporal_horizons):
            raise ValueError("temporal_horizons must not contain duplicates.")


class DecisionAdapter(nn.Module):
    """Map stable state embeddings to retrieval and utility-prediction spaces."""

    def __init__(self, config: DecisionAdapterConfig | None = None) -> None:
        super().__init__()
        self.config = config or DecisionAdapterConfig()
        self.projection = nn.Sequential(
            nn.LayerNorm(self.config.input_dim),
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.decision_dim),
        )
        self.utility_head = nn.Linear(self.config.decision_dim, len(self.config.quantiles))
        self.event_head = (
            nn.Linear(self.config.decision_dim, 1 + 2 * len(self.config.temporal_horizons))
            if self.config.temporal_horizons
            else None
        )
        if self.config.include_cash_logit:
            self.cash_logit = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("cash_logit", None)

    def _utility_quantiles(self, decision: torch.Tensor) -> torch.Tensor:
        raw = self.utility_head(decision)
        if not self.config.enforce_non_crossing_quantiles:
            return raw
        median = raw[:, 1]
        return torch.stack(
            [median - F.softplus(raw[:, 0]), median, median + F.softplus(raw[:, 2])],
            dim=1,
        )

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.projection(latent)
        decision = F.normalize(raw, p=2, dim=-1, eps=1e-4)
        output = {
            "decision": decision,
            "utility_quantiles": self._utility_quantiles(decision),
        }
        if self.event_head is not None:
            output["event_logits"] = self.event_head(decision)
        if self.cash_logit is not None:
            output["cash_logit"] = self.cash_logit
        return output

    def checkpoint(self) -> dict[str, object]:
        """Return a portable state dictionary with its exact architecture."""
        return {"config": asdict(self.config), "model_state": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, payload: dict[str, object]) -> "DecisionAdapter":
        config = DecisionAdapterConfig(**payload["config"])
        model = cls(config)
        model.load_state_dict(payload["model_state"])
        return model
