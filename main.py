"""
main.py — Lean research entry point.

  1. Load OHLCV for all tickers
  2. Detect price cycles (self-supervision signal)
  3. Print summary
"""

import logging
import torch

from src.data.loader import load_market_data
from src.cycle.cycle_detector import detect_cycles, Cycle
from src.visualization.price_cycle_plot import plot_price_and_cycles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG_PATH = "config/market_universe.yaml"
START = "2018-01-01"
END = "2024-12-31"


def run_stock(ticker: str, price_df: pd.DataFrame) -> list[Cycle]:
    close = price_df["close"]
    cycles = detect_cycles(close)

    print(f"[{ticker}] {len(cycles)} cycles detected")
    for i, c in enumerate(cycles):
        print(f"  {i+1}: {c.start_date.date()} → {c.end_date.date()} "
              f"({c.duration_days}d, {c.net_return:.1%})")

    if cycles:
        plot_price_and_cycles(
            dates=price_df.index,
            close_prices=close.values,
            cycles=cycles,
            title=f"{ticker} | Price Cycles",
        )
    return cycles


def main():
    # 1. Load
    data = load_market_data(CONFIG_PATH, START, END)
    if not data:
        logger.error("No data loaded — check config or network.")
        return

    # 2. Detect cycles
    all_cycles = {}
    for ticker, dfs in data.items():
        price_df = dfs["price"]
        if price_df.empty:
            continue
        print(f"\n--- {ticker} ---")
        all_cycles[ticker] = run_stock(ticker, price_df)

    total = sum(len(v) for v in all_cycles.values())
    print(f"\nTotal: {total} cycles across {len(all_cycles)} tickers")


if __name__ == "__main__":
    main()
