"""
Lean data loader — fetches OHLCV via yfinance with no caching overhead.
"""

import logging
import os

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)


def load_tickers_from_yaml(config_path: str) -> list[str]:
    """Extract a flat ticker list from market_universe.yaml."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    tickers = config.get("tickers", [])
    if not tickers:
        for industry in config.get("industries", {}).values():
            if isinstance(industry, dict):
                tickers.extend(industry.get("tickers", []))
    return tickers


def fetch_ohlcv(
    ticker: str, start: str, end: str, interval: str = "1d"
) -> pd.DataFrame:
    """Download OHLCV for a single ticker.  Returns empty DataFrame on failure."""
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval,
                         auto_adjust=True, progress=False)
        if df.empty:
            return df

        # yfinance >=0.2.31 returns MultiIndex columns (Price, Ticker).
        # Flatten to simple lowercase column names.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        # Ensure UTC DatetimeIndex
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        return pd.DataFrame()


def fetch_fundamentals(ticker: str) -> pd.DataFrame:
    """Download earnings history/dates for a single ticker."""
    try:
        t = yf.Ticker(ticker)
        df = t.earnings_dates
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.error(f"Failed to fetch fundamentals for {ticker}: {e}")
        return pd.DataFrame()


def load_market_data(
    config_path: str, start: str, end: str, interval: str = "1d"
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load OHLCV and Earnings for every ticker in the YAML config file."""
    tickers = load_tickers_from_yaml(config_path)
    if not tickers:
        logger.warning(f"No tickers found in {config_path}")
        return {}

    data: dict[str, dict[str, pd.DataFrame]] = {}
    for t in tickers:
        logger.info(f"Fetching {t}…")
        price_df = fetch_ohlcv(t, start, end, interval)
        earnings_df = fetch_fundamentals(t)
        
        if not price_df.empty:
            data[t] = {
                "price": price_df,
                "earnings": earnings_df
            }
        else:
            logger.warning(f"No data for {t}")
    return data
