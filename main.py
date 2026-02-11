import torch
import pandas as pd
from typing import List

from src.data.fetcher import load_market_ohlcv
# from src.data.augment import augment_3day_geometry # Not used for price cycles
from src.models.timesnet_encoder import TimesNetEncoder
from src.cycle.cycle_detector import detect_cycles, Cycle
from src.visualization.price_cycle_plot import plot_price_and_cycles

# Latent logic removed
# from src.cycle.cycle_metrics import compute_cycle_score 
# from src.visualization.latent_trajectory import plot_latent_trajectory


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_stock(
    ticker: str,
    ohlcv_df: pd.DataFrame,
) -> List[Cycle]:
    
    # Use closing prices for cycle detection
    close_prices = ohlcv_df["close"]
    
    # --- cycle detection ---
    # Using defaults: min 40 days, max 252 days, min 30% return
    cycles = detect_cycles(close_prices)
    
    print(f"[{ticker}] Detected {len(cycles)} cycles.")
    for i, c in enumerate(cycles):
        print(f"  Cycle {i+1}: {c.start_date.date()} -> {c.end_date.date()} "
              f"({c.duration_days} days, {c.net_return:.1%})")

    # --- visualization ---
    if len(cycles) > 0:
        plot_price_and_cycles(
            dates=ohlcv_df.index,
            close_prices=close_prices.values,
            cycles=cycles,
            title=f"{ticker} | Price Cycles",
        )
    else:
        print(f"[{ticker}] No cycles detected matching criteria.")

    return cycles


def main():
    # Load ALL data
    data = load_market_ohlcv(
        config_path="config/market_universe.yaml",
        start="2018-01-01",
        end="2024-12-31",
        interval="1d",
    )

    # Encoder is initialized but NOT used for detection as per requirements
    encoder = TimesNetEncoder(
        in_dim=4,
        embed_dim=128,
        num_layers=2,
        top_k=3,
        dropout=0.1,
    ).to(DEVICE)
    encoder.eval() 

    # Iterate over ALL loaded stocks
    all_cycles = {}
    
    for ticker, df in data.items():
        if df.empty:
            print(f"{ticker}: no data")
            continue

        print(f"\n--- Running {ticker} ---")
        stock_cycles = run_stock(ticker, df)
        all_cycles[ticker] = stock_cycles

    # Future: Use all_cycles for aligning TimesNet embeddings or memory construction
    # For now, we are done.


if __name__ == "__main__":
    main()

