import pandas as pd
import yfinance as yf
import os
import json
import logging
import time
import numpy as np
from typing import Optional, Dict, Any, Union
from src.data.io_utils import read_dataframe, write_dataframe
from datetime import datetime, timezone

# Setup logging
logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join("data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _ensure_utc_timestamp(ts: Any) -> str:
    """Ensure timestamp is UTC ISO string."""
    if isinstance(ts, (int, float)):
        # Assume seconds if small, millis if large? yfinance usually uses seconds or timestamps
        # But safest is pd.to_datetime
        ts = pd.to_datetime(ts, unit='s' if ts < 3e10 else 'ms', utc=True)
    elif isinstance(ts, str):
        ts = pd.to_datetime(ts, utc=True)
    elif isinstance(ts, (pd.Timestamp, datetime)):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
            
    return ts.isoformat()


def df_to_dict_safe(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}
    # df.to_dict() returns {col: {index: value}}
    d = df.to_dict()
    # sanitize keys recursively? 
    # Actually we just know the structure is likely {col: {idx: val}}
    new_d = {}
    for col, series_dict in d.items():
        col_str = str(col)
        new_series_dict = {}
        if isinstance(series_dict, dict):
            for idx, val in series_dict.items():
                new_series_dict[str(idx)] = val
            new_d[col_str] = new_series_dict
        else:
            new_d[col_str] = series_dict
    return new_d

def fetch_price_history(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Fetches price history for a ticker, utilizing a parquet cache.
    
    Args:
        ticker: Symbol string.
        start: Start date string (YYYY-MM-DD or ISO).
        end: End date string (YYYY-MM-DD or ISO).
        interval: Data interval. default "1d".
        
    Returns:
        pd.DataFrame with UTC DatetimeIndex and columns [open, high, low, close, volume].
        'close' is adjusted for splits/dividends if available (yfinance default auto_adjust=True often does this, 
        but we will ensure we use 'Close' which is usually adjusted in yfinance downloads or handle appropriately).
    """
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_prices.parquet")
    
    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)
    
    # Load cache if exists
    df_cached = None
    if os.path.exists(cache_path):
        try:
            df_cached = read_dataframe(cache_path)
            # Ensure index is datetime with UTC
            if not isinstance(df_cached.index, pd.DatetimeIndex):
                df_cached.index = pd.to_datetime(df_cached.index, utc=True)
            if df_cached.index.tz is None:
                df_cached.index = df_cached.index.tz_localize('UTC')
            else:
                df_cached.index = df_cached.index.tz_convert('UTC')
        except Exception as e:
            logger.warning(f"Failed to load cache for {ticker}: {e}")
            df_cached = None

    # Determine what needs fetching
    fetch_start = start_dt
    fetch_end = end_dt
    
    needs_fetch = True
    
    if df_cached is not None and not df_cached.empty:
        cache_start = df_cached.index.min()
        cache_end = df_cached.index.max()
        
        # If request is fully covered by cache
        if cache_start <= start_dt and cache_end >= end_dt - pd.Timedelta(days=1): 
            # (Relaxed check: if cache covers up to end_dt, we are good. 
            # Note: yfinance end is exclusive usually, but let's be safe)
            return _filter_and_format_price_df(df_cached, start_dt, end_dt)
            
        # If partial overlap or extension needed
        # Simple strategy: If we need data AFTER cache, fetch from cache_end to end_dt.
        # If we need data BEFORE cache, fetch from start_dt to cache_start.
        # For simplicity and robustness against gaps: 
        # If extended range is needed, just fetch the missing piece and merge.
        
        if end_dt > cache_end:
            fetch_start = cache_end + pd.Timedelta(days=1)
            fetch_end = end_dt
        elif start_dt < cache_start:
            fetch_start = start_dt
            fetch_end = cache_start - pd.Timedelta(days=1)
        else:
            # Should be covered, but maybe gaps?
            # If we are strictly inside, just return what we have (or re-fetch if suspicious?)
            # Prompt says "If cache exists and end <= cache_end, use cache".
            return _filter_and_format_price_df(df_cached, start_dt, end_dt)
    
    # Fetching logic with backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # yfinance download
            # auto_adjust=True gives adjusted close in 'Close' column.
            logger.info(f"Fetching {ticker} from {fetch_start} to {fetch_end}")
            df_new = yf.download(
                ticker, 
                start=fetch_start.strftime("%Y-%m-%d"), 
                end=fetch_end.strftime("%Y-%m-%d"), 
                interval=interval,
                auto_adjust=True,
                progress=False
            )
            
            if df_new.empty:
                logger.warning(f"No data found for {ticker} in range {fetch_start}-{fetch_end}")
                if df_cached is not None:
                     return _filter_and_format_price_df(df_cached, start_dt, end_dt)
                return pd.DataFrame() # Return empty if nothing found

            # Standardize columns
            df_new.columns = [c.lower() for c in df_new.columns]
            # Ensure UTC index
            if df_new.index.tz is None:
                df_new.index = df_new.index.tz_localize('UTC')
            else:
                 df_new.index = df_new.index.tz_convert('UTC')
            
            # Merge with cache
            if df_cached is not None:
                # Combine and drop duplicates
                df_combined = pd.concat([df_cached, df_new])
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                df_combined.sort_index(inplace=True)
            else:
                df_combined = df_new
            
            # Save to cache
            write_dataframe(df_combined, cache_path)
            
            return _filter_and_format_price_df(df_combined, start_dt, end_dt)
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt) # Exponential backoff
            else:
                # On final failure, return partial/cached if available or raise/empty
                if df_cached is not None:
                    return _filter_and_format_price_df(df_cached, start_dt, end_dt)
                raise e
                
    return pd.DataFrame()

def _filter_and_format_price_df(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Helper to filter info, ensure columns, and sorting."""
    # Ensure columns exist. yfinance with auto_adjust=True returns Open, High, Low, Close, Volume
    # We lowercased them.
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        # Fallback if some are missing structure
        pass
    
    # Filter range
    # Ensure start/end are tz-aware UTC matching df index
    mask = (df.index >= start) & (df.index <= end)
    df_subset = df.loc[mask].copy()
    
    return df_subset[required] if not df_subset.empty else df_subset


def fetch_fundamentals_yf(ticker: str) -> Dict[str, Any]:
    """
    Fetches fundamental data for a ticker using yfinance.
    Caches raw JSON to data/cache/{ticker}_fundamentals.json.
    """
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_fundamentals.json")
    
    # Always try to fetch fresh? Or use cache with TTL?
    # Prompt says: "Store raw JSON to data/cache...". Doesn't explicitly say "read from cache if recent".
    # But usually fundamental data doesn't change *that* often. Let's assume a 1-day TTL or forced fetch.
    # For now, let's just fetch and overwrite, or read if fetch fails.
    # Actually, efficient ingestion implies valid cache usage. Let's check update time.
    
    should_fetch = True
    if os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        if (time.time() - mtime) < 86400: # 24 hours
            should_fetch = False
            
    data = {}
    if not should_fetch:
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception:
            should_fetch = True
            
    if should_fetch:
        try:
            t = yf.Ticker(ticker)
            
            # Gather components
            # info
            info = t.info
            
            # calendar - earnings dates
            calendar = t.calendar
            # calendar is often a dict or DataFrame. Convert to dict friendly format
            if isinstance(calendar, pd.DataFrame):
                calendar = calendar.to_dict()
            elif hasattr(calendar, 'to_dict'):
                 calendar = calendar.to_dict()
            
            # quarterly earnings/financials
            # These are DataFrames usually
            q_earnings = t.quarterly_earnings
            q_financials = t.quarterly_financials


            data = {
                "info": info,
                "calendar": calendar,
                "quarterly_earnings": df_to_dict_safe(q_earnings),
                "quarterly_financials": df_to_dict_safe(q_financials),
                # "earnings_dates": df_to_dict_safe(t.earnings_dates) if hasattr(t, 'earnings_dates') else {},
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            # Try to get earnings dates specifically if possible as it holds actual report timestamps sometimes
            # t.earnings_dates usually has 'Earnings Date' index and 'EPS Estimate', 'Reported EPS', etc.
            try:
                ed = t.earnings_dates
                if ed is not None and not ed.empty:
                     # Convert index (timestamps) to string iso
                     ed_dict = {}
                     for ts, row in ed.iterrows():
                         ts_str = _ensure_utc_timestamp(ts)
                         ed_dict[ts_str] = row.to_dict()
                     data["earnings_history"] = ed_dict
            except Exception as e:
                logger.warning(f"Could not fetch earnings_dates for {ticker}: {e}")

            # Save
            with open(cache_path, 'w') as f:
                json.dump(data, f, default=str, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to fetch fundamentals for {ticker}: {e}")
            # Try to return cache if exists even if old
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    return json.load(f)
            return {}
            
    return data

def get_latest_report_before(ticker: str, as_of: pd.Timestamp) -> Dict[str, Any]:
    """
    Returns the latest earnings/revenue report whose timestamp <= as_of.
    """
    # Normalize as_of
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize('UTC')
    else:
        as_of = as_of.tz_convert('UTC')
        
    data = fetch_fundamentals_yf(ticker)
    
    # We look primarily at 'earnings_history' which comes from t.earnings_dates (specific timestamps)
    # If not available, we might assume quarterly earnings dates from financials (often just 'end of quarter' date, not report date)
    # Ideally use 'Earnings Date' from earnings_history.
    
    latest_report = {}
    best_date = None
    
    # check earnings_history
    history = data.get("earnings_history", {})
    if history:
        for date_str, report_data in history.items():
            try:
                dt = pd.to_datetime(date_str)
                if dt.tz is None:
                    dt = dt.tz_localize('UTC')
                
                if dt <= as_of:
                    if best_date is None or dt > best_date:
                        best_date = dt
                        latest_report = report_data
                        latest_report['report_date'] = date_str
            except:
                continue
                
    # If explicit history not found, fallback to quarterly_earnings (dates are usually indices?)
    if not latest_report:
        # q_earnings format in cache: {column: {index: value}} or similar depending on yf structure
        pass 
        
    return latest_report


def load_market_ohlcv(
    config_path: str,
    start: str,
    end: str,
    interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """
    Loads OHLCV data for multiple tickers from a YAML config file.
    
    Args:
        config_path: Path to YAML config file containing ticker list.
        start: Start date string (YYYY-MM-DD or ISO).
        end: End date string (YYYY-MM-DD or ISO).
        interval: Data interval, default "1d".
        
    Returns:
        Dictionary mapping ticker symbols to DataFrames with OHLCV data.
    """
    import yaml
    
    # Load config
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}. Using empty ticker list.")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}
    
    # Extract tickers
    tickers = config.get('tickers', [])
    if not tickers:
        logger.warning(f"No tickers found in config: {config_path}")
        return {}
    
    # Fetch data for each ticker
    data = {}
    for ticker in tickers:
        try:
            logger.info(f"Loading data for {ticker}")
            df = fetch_price_history(ticker, start, end, interval)
            if not df.empty:
                data[ticker] = df
            else:
                logger.warning(f"No data retrieved for {ticker}")
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            continue
    
    return data
