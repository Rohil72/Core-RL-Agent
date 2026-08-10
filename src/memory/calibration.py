from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEV_START_DATE = pd.Timestamp("2022-01-01", tz="UTC")
DEV_END_DATE = pd.Timestamp("2024-12-31", tz="UTC")


@dataclass
class CalibrationMarketSummary:
    market: str
    sample_count: int
    quantile: float
    correction: float
    fallback_used: bool
    mean_residual: float
    std_residual: float
    p50_residual: float
    p90_residual: float
    max_residual: float


@dataclass
class DownsideCalibrationModel:
    adverse_quantile: float
    min_samples: int
    start_date: str
    end_date: str
    pooled_sample_count: int
    pooled_correction: float
    market_summaries: dict[str, CalibrationMarketSummary]
    source_hash: str | None = None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the one-sided residual correction to candidate signals."""
        if df.empty:
            out = df.copy()
            out["retrieval_calibrated_downside"] = pd.Series(dtype=float)
            return out

        out = df.copy()
        pred_downside = pd.to_numeric(out.get("retrieval_expected_downside"), errors="coerce").fillna(0.0)
        predicted_loss = np.maximum(0.0, -pred_downside.to_numpy(dtype=float))

        corrections = np.zeros(len(out), dtype=float)
        if "market" in out.columns:
            for idx, market in enumerate(out["market"].astype(str)):
                if market in self.market_summaries:
                    corrections[idx] = self.market_summaries[market].correction
                else:
                    corrections[idx] = self.pooled_correction
        else:
            corrections[:] = self.pooled_correction

        calibrated_downside = -(predicted_loss + corrections)
        out["retrieval_calibrated_downside"] = calibrated_downside
        return out

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


def fit_downside_calibration(
    development_df: pd.DataFrame,
    adverse_quantile: float = 0.90,
    min_samples: int = 30,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    source_hash: str | None = None,
) -> DownsideCalibrationModel:
    """Fit a one-sided residual downside correction using ONLY development data (2022-01-01 to 2024-12-31)."""

    if development_df.empty:
        raise ValueError("Development dataset for calibration cannot be empty.")

    df = development_df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("Development dataset must contain a timestamp column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(f"{end_date} 23:59:59.999999", tz="UTC")

    # ZERO-LEAKAGE ENFORCEMENT: Strictly reject any dates outside start_date to end_date
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    if min_ts < start_ts or max_ts > end_ts:
        raise ValueError(
            f"Downside calibration data contains out-of-bounds timestamps ({min_ts} to {max_ts}). "
            f"Calibration is strictly restricted to development period {start_date} to {end_date}."
        )

    required_cols = ["retrieval_expected_downside", "decision_mae"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Calibration data missing required columns: {missing}")

    realized_loss = np.maximum(0.0, -pd.to_numeric(df["decision_mae"], errors="coerce").to_numpy(dtype=float))
    predicted_loss = np.maximum(0.0, -pd.to_numeric(df["retrieval_expected_downside"], errors="coerce").to_numpy(dtype=float))
    residuals = realized_loss - predicted_loss

    valid_mask = np.isfinite(residuals)
    valid_residuals = residuals[valid_mask]
    if len(valid_residuals) == 0:
        raise ValueError("No valid finite residuals found in development data for calibration.")

    pooled_sample_count = len(valid_residuals)
    pooled_correction = max(0.0, float(np.quantile(valid_residuals, adverse_quantile)))

    df_valid = df.iloc[valid_mask].copy()
    df_valid["residual"] = valid_residuals

    market_summaries: dict[str, CalibrationMarketSummary] = {}

    markets = df_valid["market"].astype(str).unique() if "market" in df_valid.columns else ["pooled"]
    for market in markets:
        if "market" in df_valid.columns:
            m_res = df_valid.loc[df_valid["market"].astype(str) == market, "residual"].to_numpy(dtype=float)
        else:
            m_res = valid_residuals

        sample_count = len(m_res)
        if sample_count >= min_samples:
            corr = max(0.0, float(np.quantile(m_res, adverse_quantile)))
            fallback = False
        else:
            corr = pooled_correction
            fallback = True

        market_summaries[market] = CalibrationMarketSummary(
            market=market,
            sample_count=sample_count,
            quantile=adverse_quantile,
            correction=corr,
            fallback_used=fallback,
            mean_residual=float(np.mean(m_res)),
            std_residual=float(np.std(m_res, ddof=1)) if len(m_res) > 1 else 0.0,
            p50_residual=float(np.median(m_res)),
            p90_residual=float(np.quantile(m_res, 0.90)),
            max_residual=float(np.max(m_res)),
        )

    return DownsideCalibrationModel(
        adverse_quantile=adverse_quantile,
        min_samples=min_samples,
        start_date=str(DEV_START_DATE.date()),
        end_date=str(DEV_END_DATE.date()),
        pooled_sample_count=pooled_sample_count,
        pooled_correction=pooled_correction,
        market_summaries=market_summaries,
        source_hash=source_hash,
    )
