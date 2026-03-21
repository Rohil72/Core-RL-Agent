import pytest
import pandas as pd
import numpy as np
import os
import shutil
import json
from unittest.mock import patch, MagicMock
from src.data import fetcher
from src.data.io_utils import read_dataframe, write_dataframe

# Helper to create dummy price df
def create_dummy_price_df(start, end):
    dates = pd.date_range(start, end, freq='D', tz='UTC')
    n = len(dates)
    data = {
        'open': np.linspace(100, 200, n),
        'high': np.linspace(105, 205, n),
        'low': np.linspace(95, 195, n),
        'close': np.linspace(102, 202, n), # Uptrend
        'volume': np.random.randint(1000, 10000, n)
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def mock_cache_dir(tmp_path):
    # Override CACHE_DIR in fetcher
    original_cache = fetcher.CACHE_DIR
    fetcher.CACHE_DIR = str(tmp_path / "cache")
    os.makedirs(fetcher.CACHE_DIR, exist_ok=True)
    yield fetcher.CACHE_DIR
    fetcher.CACHE_DIR = original_cache

def test_fetch_price_history_caching(mock_cache_dir):
    ticker = "TEST"
    start = "2023-01-01"
    end = "2023-01-10"
    
    # Mock yfinance download
    expected_df = create_dummy_price_df(start, end)
    
    with patch('yfinance.download') as mock_download:
        mock_download.return_value = expected_df
        
        # 1. First fetch (should call download)
        df1 = fetcher.fetch_price_history(ticker, start, end)
        assert mock_download.call_count == 1
        assert len(df1) == 10
        assert os.path.exists(os.path.join(mock_cache_dir, f"{ticker}_prices.parquet"))
        
        # 2. Second fetch (same range, should hit cache)
        df2 = fetcher.fetch_price_history(ticker, start, end)
        assert mock_download.call_count == 1 # No new call
        pd.testing.assert_frame_equal(df1, df2, check_freq=False)

def test_fetch_price_history_incremental(mock_cache_dir):
    ticker = "TEST"
    start1 = "2023-01-01"
    end1 = "2023-01-05"
    
    start2 = "2023-01-01" 
    end2 = "2023-01-10" # Requests more
    
    df_part1 = create_dummy_price_df(start1, end1)
    df_part2 = create_dummy_price_df(start1, end2) # full range
    
    # Pre-populate cache with part1
    cache_path = os.path.join(mock_cache_dir, f"{ticker}_prices.parquet")
    write_dataframe(df_part1, cache_path)
    
    with patch('yfinance.download') as mock_download:
        # Mock returning only the missing part
        # Logic in fetcher: if end > cache_end, fetch cache_end+1 to end.
        # cache_end is Jan 5. Need Jan 6 to Jan 10.
        missing_start = pd.Timestamp("2023-01-06", tz='UTC')
        missing_end = pd.Timestamp("2023-01-10", tz='UTC')
        
        df_missing = df_part2.loc[missing_start:missing_end]
        mock_download.return_value = df_missing
        
        df_full = fetcher.fetch_price_history(ticker, start2, end2)
        
        # Verify it merged correctly
        assert len(df_full) == 10
        assert df_full.index.min() == pd.Timestamp("2023-01-01", tz='UTC')
        assert df_full.index.max() == pd.Timestamp("2023-01-10", tz='UTC')
        
        # Check cache updated
        df_cached_new = read_dataframe(cache_path)
        assert len(df_cached_new) == 10

def test_fetch_fundamentals_yf(mock_cache_dir):
    ticker = "TEST_FUND"
    
    with patch('yfinance.Ticker') as mock_ticker_cls:
        mock_t = MagicMock()
        mock_ticker_cls.return_value = mock_t
        
        mock_t.info = {"sector": "Technology"}
        mock_t.calendar = {"Earnings Date": [pd.Timestamp("2023-02-01")]}
        mock_t.quarterly_earnings = pd.DataFrame({'Revenue': [100]}, index=[pd.Timestamp("2023-01-01")])
        mock_t.quarterly_financials = pd.DataFrame()
        
        # Mock earnings_dates
        mock_t.earnings_dates = pd.DataFrame(
            {'EPS Estimate': [1.0], 'Reported EPS': [1.2]}, 
            index=[pd.Timestamp("2023-02-01")]
        )
        
        data = fetcher.fetch_fundamentals_yf(ticker)
        
        assert data['info']['sector'] == "Technology"
        assert 'earnings_history' in data
        assert os.path.exists(os.path.join(mock_cache_dir, f"{ticker}_fundamentals.json"))

def test_get_latest_report(mock_cache_dir):
    ticker = "TEST_REPORT"
    # Create fake cache
    data = {
        "earnings_history": {
            "2023-01-15T00:00:00+00:00": {"EPS Estimate": 1.0, "Reported EPS": 1.2},
            "2023-04-15T00:00:00+00:00": {"EPS Estimate": 1.1, "Reported EPS": 1.0}
        }
    }
    with open(os.path.join(mock_cache_dir, f"{ticker}_fundamentals.json"), 'w') as f:
        json.dump(data, f)
        
    # Query before first report
    rep1 = fetcher.get_latest_report_before(ticker, pd.Timestamp("2023-01-01", tz='UTC'))
    assert rep1 == {}
    
    # Query after first report
    rep2 = fetcher.get_latest_report_before(ticker, pd.Timestamp("2023-02-01", tz='UTC'))
    assert rep2['Reported EPS'] == 1.2
    
    # Query after second report
    rep3 = fetcher.get_latest_report_before(ticker, pd.Timestamp("2023-05-01", tz='UTC'))
    assert rep3['Reported EPS'] == 1.0
