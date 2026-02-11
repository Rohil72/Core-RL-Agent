import yaml
import yfinance as yf
import pandas as pd
from typing import Dict, List


# ----------------------------
# Universe loading
# ----------------------------

def load_universe_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_tickers(config: dict) -> List[str]:
    tickers = []
    industries = config.get("industries", {})
    for _, block in industries.items():
        tickers.extend(block.get("tickers", []))
    return sorted(set(tickers))


# ----------------------------
# OHLCV fetching
# ----------------------------

def fetch_ohlcv(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data from yfinance.

    Returns:
        dict[ticker -> DataFrame]
        DataFrame columns: open, high, low, close, volume
    """
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    result: Dict[str, pd.DataFrame] = {}

    # yfinance special case: single ticker
    if len(tickers) == 1:
        df = raw.copy()
        df.columns = [c.lower() for c in df.columns]
        result[tickers[0]] = df.dropna()
        return result

    for ticker in tickers:
        if ticker not in raw.columns.levels[0]:
            continue

        df = raw[ticker].copy()
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna()

        if not df.empty:
            result[ticker] = df

    return result


# ----------------------------
# Public entry point
# ----------------------------

def load_market_ohlcv(
    config_path: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    config = load_universe_config(config_path)
    tickers = extract_tickers(config)

    print(f"Fetching OHLCV for {len(tickers)} tickers")

    data = fetch_ohlcv(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
    )

    print(f"Successfully fetched {len(data)} tickers")
    return data


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    ohlcv_data = load_market_ohlcv(
        config_path="market_universe.yaml",
        start="2015-01-01",
        end="2024-12-31",
        interval="1d",
    )

    # sanity check
    sample = next(iter(ohlcv_data.values()))
    print(sample.head())
