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

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.projection(latent)
        decision = F.normalize(raw, p=2, dim=-1, eps=1e-4)
        return {"decision": decision, "utility_quantiles": self.utility_head(decision)}

    def checkpoint(self) -> dict[str, object]:
        """Return a portable state dictionary with its exact architecture."""
        return {"config": asdict(self.config), "model_state": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, payload: dict[str, object]) -> "DecisionAdapter":
        config = DecisionAdapterConfig(**payload["config"])
        model = cls(config)
        model.load_state_dict(payload["model_state"])
        return model
