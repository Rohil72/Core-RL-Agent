
import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch
from src.data import feature_engineering

# Synthetic Data Fixtures
@pytest.fixture
def price_window():
    dates = pd.date_range("2023-01-01", periods=30, freq='D', tz='UTC')
    # Uptrend
    prices = np.linspace(100, 130, 30) 
    # Add some noise
    # prices += np.random.normal(0, 0.5, 30)
    data = {
        'open': prices - 1,
        'high': prices + 1,
        'low': prices - 2,
        'close': prices,
        'volume': np.full(30, 1000)
    }
    return pd.DataFrame(data, index=dates)

def test_compute_price_features(price_window):
    feats = feature_engineering.compute_price_features(price_window)
    
    # Check keys
    expected = ['momentum_3', 'momentum_10', 'momentum_21', 'vol_21', 'avg_volume_21', 'trend_slope']
    for k in expected:
        assert k in feats
        
    # Validation
    # Momentum 3: price[-1] / price[-4] - 1
    # price[-1] = 130
    # price[-4] (index 26) = 100 + (30/29)*26 = approx 126.89
    p_curr = 130
    idx_prev = 30 - 1 - 3 # 26
    p_prev = 100 + (30/29)*(idx_prev) # 100 + 1.034*26 = 126.89
    # close enough logic
    assert feats['momentum_3'] > 0
    assert feats['trend_slope'] > 0 # Uptrend

def test_compute_fundamental_features():
    current = {
        "Reported EPS": 1.5,
        "EPS Estimate": 1.0,
        "Reported Revenue": 100,
        "Revenue Estimate": 90
    }
    
    prior = {
        "Reported EPS": 1.0, # Growth 50%
        "Reported Revenue": 80
    }
    
    feats = feature_engineering.compute_fundamental_features(current, prior)
    
    assert feats['eps_surprise'] == 0.5 # (1.5 - 1.0)/1.0
    assert feats['rev_surprise'] == (100 - 90)/90
    assert feats['eps_yoy_pct'] == 0.5 # (1.5 - 1.0)/1.0
    assert feats['minervini_flag'] == 1.0

def test_assemble_features():
    # Mock data
    cycle_ex = {
        "ticker": "TEST",
        "cycle_end": "2023-01-30T00:00:00+00:00",
        "context_window": [
            {"date": "2023-01-01T00:00:00+00:00", "open": 90, "high": 105, "low": 85, "close": 100, "volume": 1000},
            {"date": "2023-01-30T00:00:00+00:00", "open": 105, "high": 115, "low": 100, "close": 110, "volume": 1000} 
        ],
        "fundamental_snapshot": {"Reported EPS": 1.0},
        "derived_features": {"duration_days": 30, "net_return": 0.1}
    }
    
    with patch('src.data.feature_engineering.get_latest_report_before') as mock_report:
        mock_report.return_value = {} # No prior
        
        feats = feature_engineering.assemble_features(cycle_ex)
        
        assert feats['ticker'] == "TEST"
        assert 'price_momentum_3' in feats # calculated from context
        # 2 points -> lag 3 is nan?
        # prices: 2 rows. lag 3 -> indices 0, 1. len=2.
        # code: if len > lag. 2 < 3. so nan.
        assert pd.isna(feats.get('price_momentum_3'))
        
        assert feats['fund_eps_actual'] == 1.0
        assert feats['cycle_net_return'] == 0.1

def test_top_k():
    # create correlated dataset
    df = pd.DataFrame({
        'feat1': np.linspace(0, 10, 100),
        'feat2': np.random.randn(100),
        'target': np.linspace(0, 10, 100) + np.random.normal(0, 0.1, 100)
    })
    
    top = feature_engineering.top_k_correlated_features(df, 'target', k=1)
    assert top[0] == 'feat1'
