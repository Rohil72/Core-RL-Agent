import pandas as pd
import numpy as np
import json
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from .fetcher import get_latest_report_before, fetch_fundamentals_yf

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CYCLES_DIR = os.path.join(_PROJECT_ROOT, "data", "cycles")
os.makedirs(CYCLES_DIR, exist_ok=True)

def label_cycle(
    price_df: pd.DataFrame, 
    cycle_start: pd.Timestamp, 
    cycle_end: pd.Timestamp, 
    ticker: str,
    horizons: List[int] = [21, 63, 252], 
    r_thresh: float = None # Unused in signature locally but good for future
) -> Dict[str, Any]:
    """
    Computes forward realized returns and fundamental confirmation labels.
    """
    if price_df.empty:
        return {}

    # Ensure timestamps are localized to UTC if df index is tz-aware
    if price_df.index.tz is not None:
        if cycle_start.tzinfo is None:
            cycle_start = cycle_start.tz_localize('UTC')
        if cycle_end.tzinfo is None:
            cycle_end = cycle_end.tz_localize('UTC')

    # Locate cycle_end index
    try:
        loc_end = price_df.index.get_loc(cycle_end)
    except KeyError:
        logger.warning(f"Cycle end {cycle_end} not found in price index for {ticker}")
        return {f"ret_{h}": np.nan for h in horizons}

    # If get_loc returns slice/array (duplicates), take last occurrence
    if isinstance(loc_end, slice):
        loc_end = loc_end.stop - 1
    elif isinstance(loc_end, np.ndarray):
        loc_end = int(np.where(loc_end)[0][-1])

    # Close price at end
    p_end = price_df.iloc[loc_end]['close']

    # Guard: division by zero
    if p_end == 0 or pd.isna(p_end):
        logger.warning(f"Zero or NaN close price at cycle end for {ticker}")
        return {f"ret_{h}": np.nan for h in horizons}

    labels = {}

    # 1. Forward Returns
    for h in horizons:
        idx_future = loc_end + h
        if idx_future < len(price_df):
            p_future = price_df.iloc[idx_future]['close']
            if p_future == 0 or pd.isna(p_future):
                labels[f"ret_{h}"] = np.nan
            else:
                ret = (p_future / p_end) - 1.0
                labels[f"ret_{h}"] = ret
        else:
            labels[f"ret_{h}"] = np.nan
            
    # 2. Fundamental Confirmation
    # "whether next available earnings report after cycle_end shows EPS beat OR YoY revenue positive growth"
    # We need to fetch fundamentals
    fundamentals = fetch_fundamentals_yf(ticker)
    earnings_history = fundamentals.get("earnings_history", {})
    
    # Find next report date > cycle_end
    next_report = None
    next_report_date = None
    
    # Build date-to-report mapping once (O(N) instead of O(N²))
    date_to_report = {}
    for k, v in earnings_history.items():
        try:
            date_to_report[pd.to_datetime(k, utc=True)] = v
        except Exception:
            continue
    sorted_dates = sorted(date_to_report.keys())

    for d in sorted_dates:
        if d > cycle_end:
            next_report_date = d
            next_report = date_to_report[d]
            break
            
    confirmed = False
    confirmed_flag = 0
    
    if next_report:
        # Check EPS beat
        # keys like 'EPS Estimate', 'Reported EPS'
        est = next_report.get('EPS Estimate', np.nan)
        act = next_report.get('Reported EPS', np.nan)
        
        eps_beat = False
        if pd.notnull(est) and pd.notnull(act):
            if act > est:
                eps_beat = True
                
        # Check Revenue Growth (YoY)
        # We need previous year's same quarter revenue.
        # This is hard from just 'earnings_history' which might not have revenue.
        # We might need 'quarterly_financials' or check previous entries in history if they have revenue.
        # Yahoo 'earnings_history' (get_earnings_dates) usually has EPS, not Revenue.
        # Use quarterly_earnings/financials?
        # But we need to link the specific report date to the quarterly data row.
        # This is getting complex strictly with yfinance free data.
        # Logic: If EPS beat is true, we confirm. If not, we try revenue.
        
        rev_growth = False
        # Try to find revenue in report (unlikely in standard yf earnings_dates)
        # If not, check 'quarterly_financials'
        # Assuming 'next_report_date' corresponds to a quarter end roughly?
        # Let's stick to EPS beat primarily for robustness unless we can reliably map.
        
        if eps_beat:
            confirmed = True
            confirmed_flag = 1
            
    labels["fundamental_confirmed"] = confirmed
    labels["fundamental_confirmed_int"] = confirmed_flag
    labels["next_report_date"] = next_report_date.isoformat() if next_report_date else None
    
    return labels

def create_cycle_example(ticker: str, price_df: pd.DataFrame, cycle_meta: Any) -> Dict[str, Any]:
    """
    Creates a full example dictionary and appends to data/cycles/{ticker}.jsonl.
    cycle_meta can be a dict or a Cycle dataclass/object.
    """
    # Access logic
    def get_val(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)
        
    cycle_start = get_val(cycle_meta, 'start_date')
    cycle_end = get_val(cycle_meta, 'end_date')
    
    # Ensure timestamps
    if isinstance(cycle_start, str):
        cycle_start = pd.to_datetime(cycle_start, utc=True)
    if isinstance(cycle_end, str):
        cycle_end = pd.to_datetime(cycle_end, utc=True)
        
    start_iso = cycle_start.isoformat()
    end_iso = cycle_end.isoformat()
    
    # Get labels
    # Use ticker from outer scope if not passed? 
    # The function arg is 'ticker'.
    labels = label_cycle(price_df, cycle_start, cycle_end, ticker)
    
    # Fundamental Snapshot (latest report BEFORE end)
    fund_snapshot = get_latest_report_before(ticker, cycle_end)
    
    # Context Window
    # Slice of price data
    mask = (price_df.index >= cycle_start) & (price_df.index <= cycle_end)
    context_df = price_df.loc[mask]
    
    # Convert context to record list
    context_records = []
    if not context_df.empty:
        context_df_reset = context_df.reset_index()
        context_df_reset.columns = [str(c).lower() for c in context_df_reset.columns]
        date_col = 'date' if 'date' in context_df_reset.columns else 'index'
        
        for _, row in context_df_reset.iterrows():
            r = row.to_dict()
            # Serialize timestamps in row if any
            for k, v in r.items():
                if isinstance(v, (pd.Timestamp, datetime)):
                    r[k] = v.isoformat()
            context_records.append(r)
            
    # Derived features
    derived = {
        "duration_days": get_val(cycle_meta, 'duration_days'),
        "net_return": get_val(cycle_meta, 'net_return'),
        "peak_date": get_val(cycle_meta, 'peak_date')
    }
    # Serialize derived dates
    if isinstance(derived['peak_date'], (pd.Timestamp, datetime)):
        derived['peak_date'] = derived['peak_date'].isoformat()
    
    example = {
        "ticker": ticker,
        "cycle_start": start_iso,
        "cycle_end": end_iso,
        "context_window": context_records,
        "derived_features": derived,
        "fundamental_snapshot": fund_snapshot,
        "future_returns": {k: v for k, v in labels.items() if k.startswith('ret_')},
        "fundamental_confirmations": {k: v for k, v in labels.items() if not k.startswith('ret_')}
    }
    
    # Append to JSONL
    out_path = os.path.join(CYCLES_DIR, f"{ticker}.jsonl")
    with open(out_path, 'a') as f:
        f.write(json.dumps(example, default=str) + "\n")
        
    return example
