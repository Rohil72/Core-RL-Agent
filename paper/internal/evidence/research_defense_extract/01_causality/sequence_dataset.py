from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class FeatureStandardizer:
    mean: dict[str, float]
    std: dict[str, float]

    def transform(self, frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        out = frame.copy()
        for col in columns:
            mean = self.mean[col]
            std = self.std[col]
            out[col] = (out[col] - mean) / std
        return out

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        columns: Sequence[str],
    ) -> "FeatureStandardizer":
        mean = {}
        std = {}
        for col in columns:
            col_mean = float(frame[col].mean())
            col_std = float(frame[col].std())
            mean[col] = col_mean
            std[col] = col_std if col_std > 1e-6 else 1.0
        return cls(mean=mean, std=std)


def select_holdout_tickers(
    tickers: Sequence[str],
    fraction: float,
    seed: int,
) -> list[str]:
    tickers = sorted(set(tickers))
    if fraction <= 0 or not tickers:
        return []
    count = max(1, int(round(len(tickers) * fraction)))
    rng = random.Random(seed)
    selected = tickers[:]
    rng.shuffle(selected)
    return sorted(selected[:count])


def build_walk_forward_splits(
    index: pd.DatetimeIndex,
    train_years: int,
    val_years: int,
    test_years: int,
    step_years: int,
) -> list[dict[str, pd.Timestamp]]:
    years = sorted(index.year.unique())
    if not years:
        return []

    min_year = int(years[0])
    max_year = int(years[-1])
    splits = []
    start_year = min_year

    while True:
        train_end_year = start_year + train_years - 1
        val_end_year = train_end_year + val_years
        test_end_year = val_end_year + test_years
        if test_end_year > max_year:
            break
        splits.append(
            {
                "train_start": pd.Timestamp(f"{start_year}-01-01", tz="UTC"),
                "train_end": pd.Timestamp(f"{train_end_year}-12-31", tz="UTC"),
                "val_start": pd.Timestamp(f"{train_end_year + 1}-01-01", tz="UTC"),
                "val_end": pd.Timestamp(f"{val_end_year}-12-31", tz="UTC"),
                "test_start": pd.Timestamp(f"{val_end_year + 1}-01-01", tz="UTC"),
                "test_end": pd.Timestamp(f"{test_end_year}-12-31", tz="UTC"),
            }
        )
        start_year += step_years

    return splits


def filter_frame_by_period(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return frame[(frame.index >= start) & (frame.index <= end)].copy()


def build_buffered_period_frame(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    history_rows: int,
) -> pd.DataFrame:
    chunks = []
    for _, group in frame.groupby("ticker"):
        g = group.sort_index()
        in_period = (g.index >= start) & (g.index <= end)
        if not in_period.any():
            continue
        positions = np.flatnonzero(np.asarray(in_period))
        start_pos = max(0, int(positions[0]) - history_rows)
        end_pos = int(positions[-1]) + 1
        chunks.append(g.iloc[start_pos:end_pos].copy())

    if not chunks:
        return frame.iloc[0:0].copy()
    return pd.concat(chunks).sort_index()


class CycleSequenceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        feature_cols: Sequence[str],
        future_target_cols: Sequence[str],
        window_size: int,
        target_start: pd.Timestamp | None = None,
        target_end: pd.Timestamp | None = None,
        max_label_horizon_sessions: int = 0,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.future_target_cols = list(future_target_cols)
        self.window_size = int(window_size)
        self.target_start = target_start
        self.target_end = target_end
        self.max_label_horizon_sessions = int(max_label_horizon_sessions)
        self.frame = frame
        self.series: dict[str, dict[str, Any]] = {}
        self.samples: list[tuple[str, int]] = []
        self.sample_markets: list[str] = []

        for ticker, group in frame.groupby("ticker"):
            g = group.sort_index().copy()
            if len(g) < self.window_size:
                continue

            feature_values = g[self.feature_cols].to_numpy(dtype=np.float32)
            action_values = g["oracle_action"].to_numpy(dtype=np.int64)
            future_values = g[self.future_target_cols].to_numpy(dtype=np.float32)
            gate_values = (
                g["tech_minervini_gate"].to_numpy(dtype=np.float32)
                if "tech_minervini_gate" in g.columns
                else np.ones(len(g), dtype=np.float32)
            )
            hard_negative_values = (
                g["oracle_hard_negative"].to_numpy(dtype=np.float32)
                if "oracle_hard_negative" in g.columns
                else np.zeros(len(g), dtype=np.float32)
            )
            close_values = g["close"].to_numpy(dtype=np.float32)

            self.series[ticker] = {
                "features": feature_values,
                "actions": action_values,
                "future": future_values,
                "gate": gate_values,
                "hard_negative": hard_negative_values,
                "close": close_values,
                "index": g.index,
                "market": str(g["market"].iloc[0]) if "market" in g else "unknown",
            }

            for end_idx in range(self.window_size - 1, len(g)):
                timestamp = g.index[end_idx]
                if self.target_start is not None and timestamp < self.target_start:
                    continue
                if self.target_end is not None and timestamp > self.target_end:
                    continue
                # Split-boundary purging: drop samples whose future label horizon exceeds boundary
                if self.max_label_horizon_sessions > 0:
                    if end_idx + self.max_label_horizon_sessions >= len(g):
                        continue
                    if self.target_end is not None:
                        # Ensure future horizon timestamp does not cross target_end
                        future_ts = g.index[end_idx + self.max_label_horizon_sessions]
                        if future_ts > self.target_end:
                            continue

                self.samples.append((ticker, end_idx))
                self.sample_markets.append(str(self.series[ticker]["market"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        ticker, end_idx = self.samples[idx]
        series = self.series[ticker]
        start_idx = end_idx - self.window_size + 1

        window = series["features"][start_idx : end_idx + 1]
        timestamp = series["index"][end_idx]
        return {
            "sequence": torch.from_numpy(window.T.copy()),
            "action": torch.tensor(series["actions"][end_idx], dtype=torch.long),
            "future_target": torch.from_numpy(series["future"][end_idx].copy()),
            "gate": torch.tensor(series["gate"][end_idx], dtype=torch.float32),
            "hard_negative": torch.tensor(
                series["hard_negative"][end_idx], dtype=torch.float32
            ),
            "close": torch.tensor(series["close"][end_idx], dtype=torch.float32),
            "ticker": ticker,
            "market": str(series["market"]),
            "timestamp": timestamp.isoformat(),
        }
