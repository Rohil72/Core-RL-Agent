"""
Lean research entry point.

1. Load OHLCV for all tickers
2. Detect price cycles
3. Print a summary
"""

import logging

import pandas as pd
import torch

from src.cycle.cycle_detector import Cycle, detect_cycles
from src.data.features import (
    compute_fundamental_features_aligned,
    compute_technical_features,
)
from src.data.loader import load_market_data
from src.visualization.price_cycle_plot import plot_price_and_cycles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG_PATH = "config/market_universe.yaml"
START = "2018-01-01"
END = "2024-12-31"
DETECTOR_KWARGS = {
    "min_duration_days": 21,
    "max_duration_days": 252,
    "min_return": 0.30,
    "soft_pullback_limit": 0.05,
    "hard_pullback_limit": 0.12,
    "volatility_window": 21,
    "volatility_multiplier": 2.0,
    "min_cycle_score": 0.58,
    "min_quality_score": 0.20,
}


def run_stock(
    ticker: str,
    price_df: pd.DataFrame,
    earnings_df: pd.DataFrame,
) -> list[Cycle]:
    feature_df = compute_technical_features(price_df)
    feature_df = compute_fundamental_features_aligned(feature_df, earnings_df)
    close = feature_df["close"]
    cycles = detect_cycles(
        close,
        feature_frame=feature_df,
        **DETECTOR_KWARGS,
    )

    print(f"[{ticker}] {len(cycles)} cycles detected")
    for i, cycle in enumerate(cycles, start=1):
        print(
            f"  {i}: {cycle.start_date.date()} -> {cycle.end_date.date()} "
            f"({cycle.duration_days}d, {cycle.net_return:.1%}, "
            f"dd={cycle.max_drawdown_from_start:.1%}, q={cycle.quality_score:.2f}, "
            f"score={cycle.cycle_score:.2f})"
        )

    if cycles:
        plot_price_and_cycles(
            dates=feature_df.index,
            close_prices=close.values,
            cycles=cycles,
            title=f"{ticker} | Price Cycles",
        )
    return cycles


def main() -> None:
    data = load_market_data(CONFIG_PATH, START, END)
    if not data:
        logger.error("No data loaded; check config or network.")
        return

    all_cycles: dict[str, list[Cycle]] = {}
    for ticker, dfs in data.items():
        price_df = dfs["price"]
        earnings_df = dfs["earnings"]
        if price_df.empty:
            continue
        print(f"\n--- {ticker} ---")
        all_cycles[ticker] = run_stock(ticker, price_df, earnings_df)

    total = sum(len(cycles) for cycles in all_cycles.values())
    print(f"\nTotal: {total} cycles across {len(all_cycles)} tickers")


if __name__ == "__main__":
    main()
