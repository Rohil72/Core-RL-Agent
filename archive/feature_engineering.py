import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.data.fetcher import get_latest_report_before
from sklearn.feature_selection import mutual_info_regression

def compute_price_features(context_df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes time-series features from a price window DataFrame.
    """
    if context_df.empty:
        return {}
        
    df = context_df.copy()
    
    # Ensure columns are lower case
    df.columns = [c.lower() for c in df.columns]
    
    # Sort by date just in case
    df.sort_index(inplace=True)
    
    # Use 'close' (or 'close_adj' if present, but prompt says 'close' in formula usually implies adjusted if doing returns)
    # The fetcher returns 'close' as adjusted if auto_adjust=True.
    prices = df['close']
    volume = df['volume']
    
    features = {}
    
    # 1. Momentum / Returns
    # Returns over 3, 10, 21 days
    # We take the LAST row and compare with T-N
    if len(prices) >= 1:
        curr_price = prices.iloc[-1]
        
        for lag in [3, 10, 21]:
            if len(prices) > lag:
                prev_price = prices.iloc[-(lag+1)]
                features[f'momentum_{lag}'] = (curr_price / prev_price) - 1.0
            else:
                features[f'momentum_{lag}'] = np.nan
                
    # 2. Volatility (21 day realized vol of log returns)
    log_rets = np.log(prices / prices.shift(1)).dropna()
    if len(log_rets) >= 21:
        # Standard deviation of last 21 returns
        vol_21_daily = log_rets.tail(21).std()
        # Annualized? Prompt says "realized vol of log returns", doesn't specify annualized. 
        # Usually vol is annualized, but let's stick to daily std dev or ask? 
        # "vol_21 (realized vol of log returns)" -> implies just std dev over window.
        features['vol_21'] = vol_21_daily
    else:
        features['vol_21'] = log_rets.std() if len(log_rets) > 1 else np.nan
        
    # 3. Volume
    if len(volume) >= 21:
        features['avg_volume_21'] = volume.tail(21).mean()
    else:
        features['avg_volume_21'] = volume.mean() if len(volume) > 0 else np.nan
        
    # 4. Volatility Z-Score
    # (current vol - avg vol) / std(vol_series)? 
    # Or (current_price - ma) / std? 
    # Usually "Volatility Z-Score" means compare current vol to historical vol distribution.
    # Let's compute rolling 21d vol over the window? 
    # If window is small (e.g. 1 cycle), we might not have enough history for a robust Z-score of Volatility itself.
    # Maybe "Price Z-Score relative to Volatility"? 
    # Given "volatility_zscore" is the name, let's assume z-score of the current vol_21 metric.
    # But if we only calculate one scalar vol_21, we can't z-score it without history.
    # Let's assume the window is long enough or we skip.
    features['volatility_zscore'] = np.nan
    if len(log_rets) >= 21:
        # Compute rolling 5-day vol over the window, then z-score the latest
        rolling_vol = log_rets.rolling(window=5, min_periods=3).std()
        rolling_vol_clean = rolling_vol.dropna()
        if len(rolling_vol_clean) > 1:
            vol_mean = rolling_vol_clean.mean()
            vol_std = rolling_vol_clean.std()
            if vol_std > 1e-10:
                features['volatility_zscore'] = (rolling_vol_clean.iloc[-1] - vol_mean) / vol_std
    
    # 5. Drawdown
    # From peak in window to current
    if len(prices) > 0:
        rolling_max = prices.cummax()
        drawdowns = (prices / rolling_max) - 1.0
        features['drawdown'] = drawdowns.iloc[-1]
        features['max_drawdown_in_window'] = drawdowns.min()
    
    # 6. Candles
    if len(df) > 0:
        up_candles = df[df['close'] > df['open']]
        features['num_up_candles'] = len(up_candles)
        
        candle_ranges = df['high'] - df['low']
        features['avg_candle_range'] = candle_ranges.mean()
        
    # 7. Trend Slope
    # OLS slope of log(close) vs time step (0, 1, 2...) normalized?
    # Normalized how? Maybe scale time to [0, 1].
    if len(prices) > 1:
        # Guard: ensure no zero/negative prices for log
        if np.any(prices.values <= 0):
            features['trend_slope'] = np.nan
        else:
            y = np.log(prices.values)
            x = np.arange(len(y))
            x_norm = x / (len(x) - 1) if len(x) > 1 else x

            if len(x_norm) > 1:
                try:
                    slope, intercept = np.polyfit(x_norm, y, 1)
                    features['trend_slope'] = slope
                except Exception:
                    features['trend_slope'] = np.nan
            else:
                features['trend_slope'] = np.nan
            
    return features


def compute_fundamental_features(fund_snapshot: Dict[str, Any], prior_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Computes fundamental features including YoY growth and surpises.
    """
    features = {}
    
    def get_val(d, key):
        v = d.get(key, np.nan)
        return v if v is not None else np.nan
        
    # Current Snapshot
    if not fund_snapshot:
        fund_snapshot = {}
        
    eps_act = get_val(fund_snapshot, 'Reported EPS')
    eps_est = get_val(fund_snapshot, 'EPS Estimate')
    
    # eps_surprise
    # (actual-est)/abs(est). If est is 0, maybe use denominator 1 or handle?
    if pd.notnull(eps_act) and pd.notnull(eps_est):
        denom = abs(eps_est)
        if denom == 0:
            denom = 1e-6 # small epsilon
        features['eps_surprise'] = (eps_act - eps_est) / denom
    else:
        features['eps_surprise'] = np.nan
        
    features['eps_actual'] = float(eps_act) if pd.notnull(eps_act) else np.nan
    features['eps_estimate'] = float(eps_est) if pd.notnull(eps_est) else np.nan
    
    # Revenue (often not in standard 'earnings_dates' from fetcher, but might be in future expansion)
    # If not present, NaN.
    # Keys might be different depending on fetcher source.
    rev_act = get_val(fund_snapshot, 'Reported Revenue') # hypothetical key
    rev_est = get_val(fund_snapshot, 'Revenue Estimate')
    
    if pd.notnull(rev_act) and pd.notnull(rev_est):
        denom = abs(rev_est)
        if denom == 0:
            denom = 1e-6
        features['rev_surprise'] = (rev_act - rev_est) / denom
    else:
        features['rev_surprise'] = np.nan
        
    features['rev_actual'] = float(rev_act) if pd.notnull(rev_act) else np.nan
    
    # Prior Snapshot Comparisons
    eps_yoy = np.nan
    rev_yoy = np.nan
    rolling2_eps = np.nan
    
    if prior_snapshot:
        prior_eps = get_val(prior_snapshot, 'Reported EPS')
        prior_rev = get_val(prior_snapshot, 'Reported Revenue')
        
        # YoY
        # Wait, prior_snapshot should be the SAME QUARTER LAST YEAR for YoY.
        # But 'prior_snapshot' arg usually implies "previous sequential report" if passed from assemble?
        # The prompt says: "Assemble... integrate prior report lookup".
        # And "rolling 2-report comparisons".
        # But "eps_yoy_pct = (eps_actual - prior_eps_actual)..." implies comparison with *prior*. 
        # Usually "YoY" means Year-Over-Year (4 quarters ago). 
        # "rolling 2-report" usually means sequential (QoQ).
        # Prompt definition: "eps_yoy_pct = (eps_actual - prior_eps_actual)..."
        # If "prior" means "previous record in time", then it's QoQ.
        # Given the variable name "eps_yoy_pct" but description "prior_eps_actual", there is ambiguity.
        # I will assume "prior" refers to the linearly previous report available (Sequentially), 
        # BUT if I strictly follow "YoY" naming, I should look 4 quarters back.
        # However, without full history in function args, I can only use what is passed.
        # Let's interpret "prior_snapshot" as the sequentially previous report, 
        # and calculate the growth rate. If the user named it "yoy", they might strictly mean "growth vs previous".
        # I'll compute `eps_growth_pct` and map it to `eps_yoy_pct` key as requested, noting the sequential nature if that's what we have.
        # Ideally `assemble_features` retrieves the actual YoY report if possible.
        
        if pd.notnull(eps_act) and pd.notnull(prior_eps):
            denom = abs(prior_eps)
            if denom == 0: denom = 1e-6
            eps_yoy = (eps_act - prior_eps) / denom
            rolling2_eps = eps_act - prior_eps
            
        if pd.notnull(rev_act) and pd.notnull(prior_rev):
            denom = abs(prior_rev)
            if denom == 0: denom = 1e-6
            rev_yoy = (rev_act - prior_rev) / denom
            
    features['eps_yoy_pct'] = eps_yoy
    features['rev_yoy_pct'] = rev_yoy
    features['rolling2_eps_change'] = rolling2_eps
    
    # Minervini Flag
    # eps_surprise > 0.05 AND rev_surprise > 0 AND eps_yoy_pct > 0
    # Handle NaNs: defaults to False
    minervini = False
    eps_surp = features.get('eps_surprise', np.nan)
    rev_surp = features.get('rev_surprise', np.nan)
    eps_yoy = features.get('eps_yoy_pct', np.nan)

    try:
        if (pd.notnull(eps_surp) and eps_surp > 0.05 and
            pd.notnull(rev_surp) and rev_surp > 0 and
            pd.notnull(eps_yoy) and eps_yoy > 0):
            minervini = True
    except (TypeError, ValueError) as e:
        import logging
        logging.getLogger(__name__).warning(f"Error computing minervini_flag: {e}")
        
    features['minervini_flag'] = 1.0 if minervini else 0.0
    
    return features


def assemble_features(cycle_example: Dict[str, Any], lookback_days: int = 252) -> Dict[str, Any]:
    """
    Combines price, fundamental, and meta features into a single flat dictionary.
    """
    ticker = cycle_example.get('ticker')
    cycle_end_str = cycle_example.get('cycle_end')
    cycle_end = pd.to_datetime(cycle_end_str, utc=True)
    
    # 1. Price Features
    # Context window is list of dicts
    context_data = cycle_example.get('context_window', [])
    prices_df = pd.DataFrame(context_data)
    if not prices_df.empty:
        # Check index
        if 'date' in prices_df.columns:
            prices_df['date'] = pd.to_datetime(prices_df['date'], utc=True)
            prices_df.set_index('date', inplace=True)
            
    price_feats = compute_price_features(prices_df)
    
    # 2. Fundamental Features
    current_snapshot = cycle_example.get('fundamental_snapshot', {})
    
    # Fetch prior report
    # We need to find the report strictly BEFORE the current snapshot's report date.
    # Current snapshot report date:
    prior_snapshot = {}
    report_date_str = current_snapshot.get('report_date')
    if report_date_str and ticker:
        report_date = pd.to_datetime(report_date_str, utc=True)
        # Look for report before this date
        prior_snapshot = get_latest_report_before(ticker, report_date - pd.Timedelta(days=1))
        
    fund_feats = compute_fundamental_features(current_snapshot, prior_snapshot)
    
    # 3. Structural/Meta Features
    derived = cycle_example.get('derived_features', {})
    meta_feats = {
        "duration_days": derived.get('duration_days'),
        "net_return": derived.get('net_return'),
    }
    
    # Flatten and Prefix
    final_features = {}
    final_features['ticker'] = ticker
    final_features['cycle_end'] = cycle_end_str
    
    for k, v in price_feats.items():
        final_features[f'price_{k}'] = v
        
    for k, v in fund_feats.items():
        final_features[f'fund_{k}'] = v
        
    for k, v in meta_feats.items():
        final_features[f'cycle_{k}'] = v
        
    return final_features

def top_k_correlated_features(
    df_features: pd.DataFrame, 
    target_col: str, 
    k: int = 5, 
    methods: List[str] = ['pearson', 'mutual_info']
) -> List[str]:
    """
    Finds top k features correlated with target.
    """
    # Drop non-numeric for correlation
    # Keep target
    if target_col not in df_features.columns:
        return []
        
    # Select numeric features only
    numeric_df = df_features.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        # Maybe target is boolean/cat? encode
        if df_features[target_col].dtype == 'object' or df_features[target_col].dtype == 'bool':
             numeric_df[target_col] = df_features[target_col].astype(float)
             
    # Fill NA?
    numeric_df = numeric_df.fillna(0) # Simple fill for selection
    
    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]
    
    # If no features
    if X.empty:
        return []
        
    scores = pd.DataFrame(index=X.columns)
    
    # Pearson
    if 'pearson' in methods:
        # absolute correlation
        corr = X.corrwith(y).abs()
        scores['pearson'] = corr
        
    # Mutual Info
    if 'mutual_info' in methods:
        try:
            # Discrete vs continuous target? 
            # Prompt says "top-k ... predict the chosen non-return success labels (fundamental_confirmation)". 
            # If target is binary, use mutual_info_classif? 
            # Or mutual_info_regression works generally? 
            # Prompt explicitly said "sklearn (mutual_info_regression)".
            mi = mutual_info_regression(X, y, random_state=42)
            scores['mutual_info'] = mi
        except Exception as e:
            # Fallback
            scores['mutual_info'] = 0.0
            
    # Rank aggregation
    # Rank 1 is best.
    # We want to minimize average rank? 
    # Or normalize scores? 
    # Let's simple average of normalized scores.
    
    # Fill correlation nans
    scores = scores.fillna(0)
    
    # Normalize min-max
    for col in scores.columns:
        mn = scores[col].min()
        mx = scores[col].max()
        if mx > mn:
            scores[col] = (scores[col] - mn) / (mx - mn)
        else:
            scores[col] = 0.0
            
    scores['mean_score'] = scores.mean(axis=1)
    
    top_k = scores.nlargest(k, 'mean_score').index.tolist()
    
    return top_k
