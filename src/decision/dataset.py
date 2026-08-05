from __future__ import annotations

from dataclasses import dataclass
import glob
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecisionDatasetConfig:
    """Causal counterfactual-outcome contract used by every Phase 5 learner."""

    horizons: tuple[int, ...] = (5, 10, 21, 42, 63)
    primary_horizon: int = 63
    stop_loss: float = 0.10
    min_hold_sessions: int = 5
    max_hold_sessions: int = 63
    exit_score_fraction: float = 0.50
    exit_min_score: float = 0.0
    adverse_excursion_weight: float = 0.50
    holding_penalty_per_session: float = 0.0001
    slippage_bps_per_side: float = 10.0
    benchmark_column: str = "future_blended_alpha_63"
    utility_return_mode: str = "fixed_horizon"
    temporal_horizons: tuple[int, ...] = ()
    temporal_barrier_mode: str = "fixed"
    temporal_upside_threshold: float = 0.20
    temporal_drawdown_threshold: float = 0.10
    temporal_volatility_lookback: int = 21
    temporal_barrier_reference_horizon: int = 21
    temporal_upside_volatility_multiplier: float = 2.0
    temporal_drawdown_volatility_multiplier: float = 1.0
    temporal_minimum_upside_threshold: float = 0.05
    temporal_minimum_drawdown_threshold: float = 0.03

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizons", tuple(self.horizons))
        object.__setattr__(self, "temporal_horizons", tuple(self.temporal_horizons))
        if self.utility_return_mode not in {"fixed_horizon", "a2_score_exit"}:
            raise ValueError("utility_return_mode must be 'fixed_horizon' or 'a2_score_exit'.")
        if self.temporal_barrier_mode not in {"fixed", "trailing_volatility"}:
            raise ValueError("temporal_barrier_mode must be 'fixed' or 'trailing_volatility'.")
        if tuple(sorted(self.temporal_horizons)) != self.temporal_horizons:
            raise ValueError("temporal_horizons must be strictly ordered.")
        if len(set(self.temporal_horizons)) != len(self.temporal_horizons):
            raise ValueError("temporal_horizons must not contain duplicates.")
        if any(horizon <= 0 for horizon in self.temporal_horizons):
            raise ValueError("temporal_horizons must be positive.")


def bounded_path_quality(
    mfe: pd.Series | np.ndarray,
    mae: pd.Series | np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Return favorable excursion as a bounded share of total excursion."""

    upside = np.maximum(
        np.nan_to_num(np.asarray(mfe, dtype=float), nan=0.0),
        0.0,
    )
    downside = np.abs(
        np.minimum(
            np.nan_to_num(np.asarray(mae, dtype=float), nan=0.0),
            0.0,
        )
    )
    denominator = upside + downside
    return np.divide(
        upside,
        denominator + float(epsilon),
        out=np.zeros_like(upside, dtype=float),
        where=denominator > 0.0,
    )


def _load_price_panel(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    raw_paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    candidates: list[Path] = []
    for value in raw_paths:
        text = str(value)
        if any(token in text for token in ("*", "?", "[")):
            candidates.extend(Path(match) for match in sorted(glob.glob(text, recursive=True)))
        else:
            candidates.append(Path(value))
    frames: list[pd.DataFrame] = []
    for path in candidates:
        frame = pd.read_parquet(path)
        if "ticker" not in frame:
            frame["ticker"] = path.stem
        if "timestamp" not in frame:
            frame = frame.reset_index().rename(columns={frame.index.name or "index": "timestamp"})
        close_col = next((c for c in ("close", "Close", "adj_close") if c in frame), None)
        if close_col is None:
            raise ValueError(f"No close price in {path}.")
        frames.append(frame[["ticker", "timestamp", close_col]].rename(columns={close_col: "close"}))
    if not frames:
        raise ValueError("No precomputed price files matched the decision dataset configuration.")
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    panel["ticker"] = panel["ticker"].astype(str)
    return panel.sort_values(["ticker", "timestamp"]).drop_duplicates(["ticker", "timestamp"])


def _temporal_barriers(
    prices: pd.DataFrame,
    position: int,
    config: DecisionDatasetConfig,
) -> tuple[float, float]:
    """Return ex-ante upside and downside magnitudes for temporal event labels."""

    if config.temporal_barrier_mode == "fixed":
        return config.temporal_upside_threshold, config.temporal_drawdown_threshold
    start = max(0, position - config.temporal_volatility_lookback)
    history = prices.iloc[start : position + 1]["close"].to_numpy(dtype=float)
    log_returns = np.diff(np.log(history)) if len(history) > 1 else np.empty(0)
    daily_volatility = float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else 0.0
    if not np.isfinite(daily_volatility):
        daily_volatility = 0.0
    scale = daily_volatility * np.sqrt(config.temporal_barrier_reference_horizon)
    upside = max(
        config.temporal_minimum_upside_threshold,
        config.temporal_upside_volatility_multiplier * scale,
    )
    downside = max(
        config.temporal_minimum_drawdown_threshold,
        config.temporal_drawdown_volatility_multiplier * scale,
    )
    return float(upside), float(downside)


def temporal_event_class(
    path_returns: np.ndarray,
    horizons: tuple[int, ...],
    upside_threshold: float,
    drawdown_threshold: float,
) -> tuple[int, int, str]:
    """Encode the first rally/drawdown event into a discrete competing-risk class."""

    if not horizons:
        return 0, 0, "disabled"
    upside_hits = np.flatnonzero(path_returns >= upside_threshold)
    downside_hits = np.flatnonzero(path_returns <= -drawdown_threshold)
    upside_offset = int(upside_hits[0] + 1) if len(upside_hits) else None
    downside_offset = int(downside_hits[0] + 1) if len(downside_hits) else None
    if upside_offset is None and downside_offset is None:
        return 0, 0, "none"
    if upside_offset is not None and (downside_offset is None or upside_offset < downside_offset):
        event_type, offset, type_offset = "upside", upside_offset, 1
    else:
        event_type, offset, type_offset = "drawdown", int(downside_offset), 1 + len(horizons)
    bucket = int(np.searchsorted(np.asarray(horizons), offset, side="left"))
    if bucket >= len(horizons):
        return 0, 0, "none"
    return type_offset + bucket, offset, event_type


def _attach_path_outcomes(frame: pd.DataFrame, panel: pd.DataFrame, cfg: DecisionDatasetConfig) -> pd.DataFrame:
    price_groups = {ticker: group.reset_index(drop=True) for ticker, group in panel.groupby("ticker", sort=False)}
    records: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        ticker = str(row.ticker)
        ts = pd.Timestamp(row.timestamp)
        prices = price_groups.get(ticker)
        record: dict[str, object] = {}
        if prices is None:
            records.append(record)
            continue
        pos = prices["timestamp"].searchsorted(ts)
        if pos >= len(prices) or prices.iloc[pos]["timestamp"] != ts:
            records.append(record)
            continue
        entry = float(prices.iloc[pos]["close"])
        next_pos = pos + 1
        record["decision_return_1"] = (
            float(prices.iloc[next_pos]["close"] / entry - 1.0) if next_pos < len(prices) else np.nan
        )
        for horizon in cfg.horizons:
            end = pos + horizon
            record[f"decision_return_{horizon}"] = (
                float(prices.iloc[end]["close"] / entry - 1.0) if end < len(prices) else np.nan
            )
        end = min(pos + cfg.primary_horizon, len(prices) - 1)
        maximum_temporal_horizon = max(cfg.temporal_horizons, default=0)
        maturity_pos = pos + max(cfg.max_hold_sessions, maximum_temporal_horizon)
        record["decision_outcome_available_timestamp"] = (
            prices.iloc[maturity_pos]["timestamp"] if maturity_pos < len(prices) else pd.NaT
        )
        path = prices.iloc[pos + 1 : end + 1]["close"].to_numpy(dtype=float) / entry - 1.0
        record["decision_mfe"] = float(np.max(path)) if path.size else np.nan
        record["decision_mae"] = float(np.min(path)) if path.size else np.nan
        record["decision_path_quality"] = (
            float(np.max(path) / (abs(np.min(path)) + 1e-6)) if path.size else np.nan
        )
        if cfg.temporal_horizons:
            temporal_end = min(pos + maximum_temporal_horizon, len(prices) - 1)
            temporal_path = (
                prices.iloc[pos + 1 : temporal_end + 1]["close"].to_numpy(dtype=float) / entry - 1.0
            )
            upside_barrier, downside_barrier = _temporal_barriers(prices, pos, cfg)
            if len(temporal_path) < maximum_temporal_horizon:
                event_class, event_offset, event_type = np.nan, np.nan, "immature"
            else:
                event_class, event_offset, event_type = temporal_event_class(
                    temporal_path,
                    cfg.temporal_horizons,
                    upside_barrier,
                    downside_barrier,
                )
            record["decision_event_class"] = event_class
            record["decision_event_offset"] = event_offset
            record["decision_event_type"] = event_type
            record["decision_event_upside_barrier"] = upside_barrier
            record["decision_event_drawdown_barrier"] = downside_barrier
        records.append(record)
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def _attach_a2_counterfactual(frame: pd.DataFrame, signals: pd.DataFrame | None, cfg: DecisionDatasetConfig) -> pd.DataFrame:
    frame = frame.copy()
    frame["decision_a2_holding_sessions"] = cfg.max_hold_sessions
    frame["decision_a2_exit_reason"] = "max_hold"
    if signals is None or "opportunity_score" not in signals:
        return frame
    signal_cols = ["ticker", "timestamp", "opportunity_score"]
    available = [c for c in ("close", "retrieval_ood_pass") if c in signals]
    signals = signals[signal_cols + available].copy()
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)
    signals["ticker"] = signals["ticker"].astype(str)
    groups = {ticker: part.sort_values("timestamp").reset_index(drop=True) for ticker, part in signals.groupby("ticker")}
    holds: list[int] = []
    reasons: list[str] = []
    returns: list[float] = []
    for row in frame.itertuples(index=False):
        group = groups.get(str(row.ticker))
        if group is None:
            holds.append(cfg.max_hold_sessions); reasons.append("max_hold"); returns.append(np.nan); continue
        pos = group["timestamp"].searchsorted(pd.Timestamp(row.timestamp))
        if pos >= len(group) or group.iloc[pos]["timestamp"] != pd.Timestamp(row.timestamp):
            holds.append(cfg.max_hold_sessions); reasons.append("max_hold"); returns.append(np.nan); continue
        entry_score = group.iloc[pos]["opportunity_score"]
        entry_price = group.iloc[pos].get("close", np.nan)
        threshold = max(cfg.exit_min_score, float(entry_score) * cfg.exit_score_fraction) if pd.notna(entry_score) else cfg.exit_min_score
        exit_pos = min(pos + cfg.max_hold_sessions, len(group) - 1)
        reason = "max_hold"
        for candidate in range(pos + cfg.min_hold_sessions, exit_pos + 1):
            current = group.iloc[candidate]
            if pd.notna(entry_price) and pd.notna(current.get("close", np.nan)):
                if float(current["close"] / entry_price - 1.0) <= -cfg.stop_loss:
                    exit_pos, reason = candidate, "stop_loss"; break
            if pd.notna(current["opportunity_score"]) and float(current["opportunity_score"]) <= threshold:
                exit_pos, reason = candidate, "score_exit"; break
        holds.append(max(exit_pos - pos, 1)); reasons.append(reason)
        exit_price = group.iloc[exit_pos].get("close", np.nan)
        returns.append(float(exit_price / entry_price - 1.0) if pd.notna(entry_price) and pd.notna(exit_price) else np.nan)
    frame["decision_a2_holding_sessions"] = holds
    frame["decision_a2_exit_reason"] = reasons
    frame["decision_a2_return"] = returns
    return frame


def build_decision_frame(
    latent_frame: pd.DataFrame,
    precomputed_paths: str | Path | Iterable[str | Path],
    config: DecisionDatasetConfig | None = None,
    a2_signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build leakage-auditable decision outcomes from frozen states and later prices."""
    cfg = config or DecisionDatasetConfig()
    required = {"ticker", "timestamp"}
    if missing := required.difference(latent_frame.columns):
        raise ValueError(f"Latent frame is missing required columns: {sorted(missing)}")
    out = latent_frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["ticker"] = out["ticker"].astype(str)
    out = _attach_path_outcomes(out, _load_price_panel(precomputed_paths), cfg)
    out = _attach_a2_counterfactual(out, a2_signals, cfg)
    primary = f"decision_return_{cfg.primary_horizon}"
    if cfg.utility_return_mode == "a2_score_exit" and "decision_a2_return" in out:
        realized = out["decision_a2_return"].fillna(out[primary])
        holding = out["decision_a2_holding_sessions"]
        exit_reason = out["decision_a2_exit_reason"]
    else:
        realized = out[primary]
        holding = pd.Series(cfg.primary_horizon, index=out.index, dtype=float)
        exit_reason = pd.Series("fixed_horizon", index=out.index, dtype=object)
    out["decision_holding_sessions"] = holding
    out["decision_exit_reason"] = exit_reason
    if cfg.benchmark_column in out:
        benchmark_alpha = out[cfg.benchmark_column].astype(float)
        benchmark_return = out[primary].astype(float) - benchmark_alpha
    else:
        benchmark_return = out.groupby("timestamp")[primary].transform("median")
    costs = 2.0 * cfg.slippage_bps_per_side / 10_000.0
    out["decision_benchmark_return"] = benchmark_return
    out["decision_net_alpha"] = realized - benchmark_return - costs
    out["decision_utility"] = (
        out["decision_net_alpha"]
        - cfg.adverse_excursion_weight * out["decision_mae"].abs()
        - cfg.holding_penalty_per_session * out["decision_holding_sessions"]
    )
    out["decision_same_date_rank"] = out.groupby("timestamp")["decision_utility"].rank(pct=True)
    out["decision_is_mature"] = out["decision_outcome_available_timestamp"].notna()
    return out
