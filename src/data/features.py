"""
Feature engineering for technical and fundamental indicators.
Cleaned and merged from archived modules.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily technical features from OHLCV.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # 1. Momentum (3d, 10d, 21d)
    for lag in [3, 10, 21]:
        df[f'tech_momentum_{lag}'] = df['close'].pct_change(lag)
        
    # 2. Volatility (21d rolling realized vol)
    df['tech_vol_21'] = df['close'].pct_change().rolling(window=21).std()
    
    # 3. Volume Trend (MA 21)
    df['tech_avg_volume_21'] = df['volume'].rolling(window=21).mean()
    df['tech_volume_ratio'] = df['volume'] / (df['tech_avg_volume_21'] + 1e-9)
    
    # 4. Drawdown from 252d Peak
    rolling_max = df['close'].rolling(window=252, min_periods=1).max()
    df['tech_drawdown'] = (df['close'] / rolling_max) - 1.0
    
    # 5. Trend Slope (10d price slope)
    def calculate_slope(x):
        if len(x) < 10 or np.any(np.isnan(x)): return 0.0
        return np.polyfit(np.arange(len(x)), x, 1)[0]
    
    df['tech_trend_slope'] = df['close'].rolling(window=10).apply(calculate_slope)
    
    return df

def compute_fundamental_features_aligned(
    price_df: pd.DataFrame, 
    earnings_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aligns fundamental reports with price data, ensuring no leakage.
    Adds EPS surprises and YoY growth trends.
    """
    # 1. Precalculate metrics in earnings_df
    # earnings_df should have [Report Date, Reported EPS, EPS Estimate]
    edf = earnings_df.copy()
    edf['report_date'] = pd.to_datetime(edf.iloc[:, 0], utc=True) # Assume 1st col is date
    edf = edf.sort_values('report_date')
    edf = edf.dropna(subset=['report_date'])
    
    if edf.empty:
        # If no valid report dates, return price_df with dummy cols
        for col in ['fund_eps_surprise', 'fund_eps_growth_yoy', 'fund_eps_accel']:
            price_df[col] = 0.0
        return price_df
    denom = edf['EPS Estimate'].abs().fillna(1e-6).replace(0, 1e-6)
    edf['fund_eps_surprise'] = (edf['Reported EPS'] - edf['EPS Estimate']) / denom
    
    # YoY Growth Trend (Difference in surprises/actuals)
    edf['fund_eps_growth_yoy'] = edf['Reported EPS'].pct_change(periods=4) # True YoY if quarterly
    edf['fund_eps_accel'] = edf['fund_eps_growth_yoy'].diff() # Directionality
    
    # 2. Join with price_df
    # We use a merge_asof to ensure we only see reports *before* the price date
    price_df = price_df.sort_index()
    # Align on index
    price_with_date = price_df.reset_index()
    date_col = price_with_date.columns[0]
    
    merged = pd.merge_asof(
        price_with_date,
        edf[['report_date', 'fund_eps_surprise', 'fund_eps_growth_yoy', 'fund_eps_accel']],
        left_on=date_col,
        right_on='report_date',
        direction='backward'
    )
    
    return merged.set_index(date_col)
