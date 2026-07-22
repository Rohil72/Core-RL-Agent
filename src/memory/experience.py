from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperienceSchema:
    """Column mapping used to build reusable historical memory experiences."""

    target_upside: str = "future_max_return_63"
    target_alpha: str | None = None
    target_downside: str = "future_min_return_63"
    target_path_quality: str = "event_upside_before_drawdown_126"
    target_holding_period: str | None = "event_peak_offset_63"


@dataclass(frozen=True)
class MarketExperience:
    """One historical market state plus its realized future outcome."""

    index: int
    latent: np.ndarray
    ticker: str
    timestamp: pd.Timestamp
    upside: float
    downside: float
    alpha: float | None = None
    path_quality: float | None = None
    holding_period: float | None = None
    sector: str | None = None
    industry: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExperienceMemory:
    """Queryable in-memory database of historical market experiences."""

    frame: pd.DataFrame
    latent_matrix: np.ndarray
    latent_cols: list[str]
    schema: ExperienceSchema
    upside_col: str
    alpha_col: str | None
    downside_col: str
    path_quality_col: str | None
    holding_period_col: str | None

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, schema: ExperienceSchema | None = None) -> "ExperienceMemory":
        """Build a memory database from a latent dataframe."""
        schema = schema or ExperienceSchema()
        out = frame.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out["ticker"] = out["ticker"].astype(str)
        cols = latent_columns(out)
        upside_col = resolve_column(out, schema.target_upside)
        alpha_col = resolve_column(out, schema.target_alpha) if schema.target_alpha else None
        downside_col = resolve_column(out, schema.target_downside)
        path_col = resolve_column(out, schema.target_path_quality)
        hold_col = resolve_column(out, schema.target_holding_period) if schema.target_holding_period else None
        if upside_col is None or downside_col is None:
            raise ValueError("Memory table must contain upside and downside realized outcomes.")
        if schema.target_alpha is not None and alpha_col is None:
            raise ValueError(f"Memory table must contain configured alpha target {schema.target_alpha!r}.")
        out = out.sort_values("timestamp").reset_index(drop=True)
        matrix = out[cols].to_numpy(dtype=float)
        return cls(out, matrix, cols, schema, upside_col, alpha_col, downside_col, path_col, hold_col)

    def row_to_experience(self, row_index: int) -> MarketExperience:
        """Convert a row index into a typed historical experience."""
        row = self.frame.iloc[int(row_index)]
        return MarketExperience(
            index=int(row_index),
            latent=self.latent_matrix[int(row_index)],
            ticker=str(row["ticker"]),
            timestamp=pd.Timestamp(row["timestamp"]),
            upside=_float_or_nan(row[self.upside_col]),
            downside=_float_or_nan(row[self.downside_col]),
            alpha=_optional_float(row[self.alpha_col]) if self.alpha_col else None,
            path_quality=_optional_float(row[self.path_quality_col]) if self.path_quality_col else None,
            holding_period=_optional_float(row[self.holding_period_col]) if self.holding_period_col else None,
            sector=str(row["sector"]) if "sector" in row and pd.notna(row["sector"]) else None,
            industry=str(row["industry"]) if "industry" in row and pd.notna(row["industry"]) else None,
        )


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _optional_float(value: Any) -> float | None:
    try:
        out = float(value)
        return out if np.isfinite(out) else None
    except Exception:
        return None


def latent_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("latent_")]
    cols = sorted(cols, key=lambda c: int(c.split("_", 1)[1]) if c.split("_", 1)[1].isdigit() else c)
    if not cols:
        raise ValueError("No latent_* columns found.")
    return cols


def resolve_column(df: pd.DataFrame, base: str | None) -> str | None:
    if base is None:
        return None
    candidates = [
        base,
        f"true_{base}",
        f"target_{base}",
        f"pred_{base}",
        base.replace("future_", "true_future_"),
    ]
    for col in candidates:
        if col in df.columns:
            return col
    suffix_matches = [c for c in df.columns if c.endswith(base)]
    return suffix_matches[0] if suffix_matches else None
