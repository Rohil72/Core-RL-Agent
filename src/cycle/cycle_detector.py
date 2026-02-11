import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Cycle:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    start_idx: int
    end_idx: int
    duration_days: int
    net_return: float
    peak_date: pd.Timestamp
    peak_idx: int


def detect_cycles(
    prices: pd.Series,
    min_duration_days: int = 40,
    max_duration_days: int = 252,
    min_return: float = 0.30,
) -> List[Cycle]:
    """
    Detects price cycles based on duration and return constraints.
    
    Args:
        prices: pandas Series of closing prices with DatetimeIndex
        min_duration_days: Minimum duration of a cycle in calendar days (approx)
                        or trading days depending on index. 
                        We will use list index difference for simplicity (trading days).
        max_duration_days: Maximum duration (trading days).
        min_return: Minimum net return (end_price / start_price - 1).

    Returns:
        List of Cycle objects (non-overlapping).
    """
    # --- Input Validation ---
    if prices.empty:
        return []
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must have a DatetimeIndex")
    if min_duration_days <= 0 or max_duration_days <= 0:
        raise ValueError("Duration parameters must be positive")
    if min_duration_days > max_duration_days:
        raise ValueError("min_duration_days must be <= max_duration_days")
    if min_return < 0:
        raise ValueError("min_return must be non-negative")

    price_values = prices.values
    dates = prices.index
    n = len(prices)

    if n <= min_duration_days:
        return []

    candidates = []

    # O(N * MaxDuration) - deterministic search
    # We iterate every day as a potential start
    for t_start in range(n - min_duration_days):
        p_start = price_values[t_start]

        # Guard: skip zero/negative start prices
        if p_start <= 0:
            continue
        
        # Optimization: Only consider local minima or just flat iteration?
        # User asked for "deterministic logic", "volatility must be present".
        # Let's just check all valid windows first.

        # Look ahead
        for duration in range(min_duration_days, max_duration_days + 1):
            t_end = t_start + duration
            if t_end >= n:
                break
            
            p_end = price_values[t_end]
            ret = (p_end / p_start) - 1.0

            if ret >= min_return:
                # --- Structural Constraints ---
                
                # 1. Check for "pump and dump" / sharp collapse? 
                # The user says "If a detected cycle exceeds max duration, truncate".
                # My loop implicitly handles truncation by checking all lengths.
                
                # 2. "Pullbacks allowed, but price must not fully revert back near cycle start"
                # Let's check intermediate prices.
                valid_structure = True
                window_prices = price_values[t_start+1 : t_end] # exclusive of start/end
                
                if len(window_prices) > 0:
                     # Check if any price dropped below start price (full reversion)
                     # User said "near cycle start", let's be strict: cannot go below p_start * 0.98?
                     # Or just strictly < p_start.
                     if np.any(window_prices < p_start):
                         valid_structure = False
                
                if valid_structure:
                    # Compute actual peak within the cycle window
                    # (Stage 2 markup: end_date is the cycle boundary,
                    #  but the true highest price may occur before end)
                    actual_peak_idx = t_start + np.argmax(price_values[t_start:t_end + 1])
                    candidates.append(Cycle(
                        start_date=dates[t_start],
                        end_date=dates[t_end],
                        start_idx=t_start,
                        end_idx=t_end,
                        duration_days=duration,
                        net_return=ret,
                        peak_date=dates[actual_peak_idx],
                        peak_idx=actual_peak_idx
                    ))
    
    # --- Resolve Overlaps ---
    # Strategy: Sort by Duration DESC, then Return DESC.
    # Greedy selection.
    
    # Sort candidates
    candidates.sort(key=lambda c: (c.duration_days, c.net_return), reverse=True)
    
    final_cycles = []
    occupied_indices = np.zeros(n, dtype=bool)

    for cand in candidates:
        # Check if this cycle overlaps with accepted cycles
        # Note: indices are [start, end].
        # User said "prefer longer duration".
        
        # Check overlap
        if not np.any(occupied_indices[cand.start_idx : cand.end_idx + 1]):
            final_cycles.append(cand)
            occupied_indices[cand.start_idx : cand.end_idx + 1] = True
            
    # Re-sort final cycles by time
    final_cycles.sort(key=lambda c: c.start_idx)
    
    return final_cycles

