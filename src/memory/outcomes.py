from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.memory.experience import resolve_column


@dataclass(frozen=True)
class RelativeOutcomeConfig:
    """Configuration for cross-sectional future-return outcomes."""

    return_target: str = "future_return_63"
    sector_column: str = "sector"
    minimum_sector_observations: int = 3
    universe_alpha_column: str = "future_universe_alpha_63"
    blended_alpha_column: str = "future_blended_alpha_63"
    sector_weight: float = 0.50
    group_column: str | None = None


def attach_relative_outcomes(
    frame: pd.DataFrame,
    config: RelativeOutcomeConfig | None = None,
) -> pd.DataFrame:
    """Attach universe- and sector-relative returns without changing row order."""
    cfg = config or RelativeOutcomeConfig()
    if not 0.0 <= cfg.sector_weight <= 1.0:
        raise ValueError("sector_weight must be between zero and one.")
    if cfg.minimum_sector_observations <= 0:
        raise ValueError("minimum_sector_observations must be positive.")
    if "timestamp" not in frame:
        raise ValueError("Relative outcomes require a timestamp column.")

    return_col = resolve_column(frame, cfg.return_target)
    if return_col is None:
        raise ValueError(f"Relative outcomes require {cfg.return_target!r}.")

    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    returns = pd.to_numeric(out[return_col], errors="coerce")
    universe_groups: list[pd.Series] = [out["timestamp"]]
    if cfg.group_column is not None:
        if cfg.group_column not in out:
            raise ValueError(
                f"Relative outcome group column {cfg.group_column!r} is unavailable."
            )
        universe_groups.append(out[cfg.group_column].fillna("Unknown").astype(str))
    universe = returns.groupby(universe_groups).transform("median")

    if cfg.sector_column in out:
        groups = [
            *universe_groups,
            out[cfg.sector_column].fillna("Unknown").astype(str),
        ]
        sector = returns.groupby(groups).transform("median")
        sector_count = returns.groupby(groups).transform("count")
        sector = sector.where(sector_count >= cfg.minimum_sector_observations, universe)
    else:
        sector = universe

    benchmark = cfg.sector_weight * sector + (1.0 - cfg.sector_weight) * universe
    out[cfg.universe_alpha_column] = returns - universe
    out[cfg.blended_alpha_column] = returns - benchmark
    out["future_sector_benchmark_return_63"] = sector
    out["future_universe_benchmark_return_63"] = universe
    out["relative_outcome_valid"] = np.isfinite(returns) & np.isfinite(universe)
    return out
