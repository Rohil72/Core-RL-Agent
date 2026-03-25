"""
Lean data loader for OHLCV and report-level fundamentals.
"""

import logging

import numpy as np
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
    """Download OHLCV for a single ticker. Returns empty DataFrame on failure."""
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return df

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.error("Failed to fetch %s: %s", ticker, e)
        return pd.DataFrame()


def _normalize_earnings_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["report_date", "eps_estimate", "reported_eps", "surprise_pct"]
        )

    edf = df.copy()
    if isinstance(edf.index, pd.DatetimeIndex):
        index_name = edf.index.name or "index"
        edf = edf.reset_index().rename(columns={index_name: "report_date"})
    elif "report_date" not in edf.columns:
        for col in edf.columns:
            if "date" in str(col).lower():
                edf = edf.rename(columns={col: "report_date"})
                break

    if "report_date" not in edf.columns:
        edf["report_date"] = pd.to_datetime(edf.iloc[:, 0], utc=True, errors="coerce")
    else:
        edf["report_date"] = pd.to_datetime(
            edf["report_date"], utc=True, errors="coerce"
        )

    edf = edf.rename(
        columns={
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "reported_eps",
            "Surprise(%)": "surprise_pct",
        }
    )

    for col in ["eps_estimate", "reported_eps", "surprise_pct"]:
        if col not in edf.columns:
            edf[col] = np.nan

    return (
        edf[["report_date", "eps_estimate", "reported_eps", "surprise_pct"]]
        .dropna(subset=["report_date"])
        .sort_values("report_date")
        .drop_duplicates(subset=["report_date"], keep="last")
    )


def _extract_quarterly_revenue(income_stmt: pd.DataFrame) -> pd.DataFrame:
    if income_stmt is None or income_stmt.empty:
        return pd.DataFrame(columns=["quarter_end", "total_revenue"])

    revenue = None
    for label in ["Total Revenue", "Operating Revenue"]:
        if label in income_stmt.index:
            series = income_stmt.loc[label]
            revenue = series if revenue is None else revenue.combine_first(series)

    if revenue is None:
        return pd.DataFrame(columns=["quarter_end", "total_revenue"])

    rdf = revenue.rename("total_revenue").rename_axis("quarter_end").reset_index()
    rdf["quarter_end"] = pd.to_datetime(rdf["quarter_end"], utc=True, errors="coerce")
    rdf["total_revenue"] = pd.to_numeric(rdf["total_revenue"], errors="coerce")

    return (
        rdf.dropna(subset=["quarter_end"])
        .sort_values("quarter_end")
        .drop_duplicates(subset=["quarter_end"], keep="last")
    )


def fetch_fundamentals(ticker: str) -> pd.DataFrame:
    """Download normalized earnings history plus quarterly revenue when available."""
    try:
        ticker_obj = yf.Ticker(ticker)
        earnings_df = _normalize_earnings_dates(ticker_obj.earnings_dates)
        revenue_df = _extract_quarterly_revenue(ticker_obj.quarterly_income_stmt)

        if earnings_df.empty:
            return earnings_df

        if revenue_df.empty:
            earnings_df["quarter_end"] = pd.NaT
            earnings_df["total_revenue"] = np.nan
            return earnings_df

        return pd.merge_asof(
            earnings_df.sort_values("report_date"),
            revenue_df.sort_values("quarter_end"),
            left_on="report_date",
            right_on="quarter_end",
            direction="backward",
            tolerance=pd.Timedelta(days=120),
        )
    except Exception as e:
        logger.error("Failed to fetch fundamentals for %s: %s", ticker, e)
        return pd.DataFrame()


def load_market_data(
    config_path: str, start: str, end: str, interval: str = "1d"
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load OHLCV and report history for every ticker in the YAML config file."""
    tickers = load_tickers_from_yaml(config_path)
    if not tickers:
        logger.warning("No tickers found in %s", config_path)
        return {}

    data: dict[str, dict[str, pd.DataFrame]] = {}
    for ticker in tickers:
        logger.info("Fetching %s...", ticker)
        price_df = fetch_ohlcv(ticker, start, end, interval)
        fundamentals_df = fetch_fundamentals(ticker)

        if not price_df.empty:
            data[ticker] = {
                "price": price_df,
                "earnings": fundamentals_df,
            }
        else:
            logger.warning("No data for %s", ticker)
    return data
