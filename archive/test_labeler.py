import pytest
import pandas as pd
import numpy as np
import os
import shutil
import json
from unittest.mock import patch, MagicMock
from src.data import labeler
from src.cycle.cycle_detector import Cycle

@pytest.fixture
def mock_cycles_dir(tmp_path):
    original_dir = labeler.CYCLES_DIR
    labeler.CYCLES_DIR = str(tmp_path / "cycles")
    os.makedirs(labeler.CYCLES_DIR, exist_ok=True)
    yield labeler.CYCLES_DIR
    labeler.CYCLES_DIR = original_dir

def create_price_df():
    dates = pd.date_range("2023-01-01", "2024-01-01", freq='D', tz='UTC')
    # Constant growth 1% per day
    prices = 100 * (1.01 ** np.arange(len(dates)))
    data = {
        'close': prices,
        'open': prices, 
        'high': prices, 
        'low': prices, 
        'volume': 1000
    }
    return pd.DataFrame(data, index=dates)

def test_label_cycle_returns():
    df = create_price_df()
    ticker = "TEST"
    
    # Cycle from Jan 1 to Jan 20
    cycle_start = df.index[0]
    cycle_end = df.index[20] # Day 20
    
    with patch('src.data.labeler.fetch_fundamentals_yf') as mock_fetch:
        mock_fetch.return_value = {} # No fundamentals
        
        labels = labeler.label_cycle(df, cycle_start, cycle_end, ticker, horizons=[1, 5])
        
        # Price at 20: 100 * 1.01^20
        # Price at 21: 100 * 1.01^21
        # Ret_1 = 1.01 - 1 = 0.01
        
        assert labels['ret_1'] == pytest.approx(0.01)
        assert labels['ret_5'] == pytest.approx((1.01**5) - 1)
        assert labels['fundamental_confirmed'] == False

def test_label_cycle_fundamentals():
    df = create_price_df()
    ticker = "TEST"
    cycle_end = df.index[20] # 2023-01-21
    
    # Mock fundamentals with a report on Feb 1 2023
    mock_data = {
        "earnings_history": {
            "2023-02-01T00:00:00+00:00": {"EPS Estimate": 1.0, "Reported EPS": 1.5} # Beat
        }
    }
    
    with patch('src.data.labeler.fetch_fundamentals_yf') as mock_fetch:
        mock_fetch.return_value = mock_data
        
        labels = labeler.label_cycle(df, df.index[0], cycle_end, ticker)
        
        assert labels['fundamental_confirmed'] == True
        assert labels['next_report_date'] == "2023-02-01T00:00:00+00:00"

def test_create_cycle_example_with_dataclass(mock_cycles_dir):
    df = create_price_df()
    ticker = "TEST_CYCLE"
    
    c = Cycle(
        start_date=df.index[0],
        end_date=df.index[10],
        start_idx=0,
        end_idx=10,
        duration_days=10,
        net_return=0.1,
        peak_date=df.index[10],
        peak_idx=10
    )
    
    with patch('src.data.labeler.fetch_fundamentals_yf') as mock_fetch:
        mock_fetch.return_value = {}
        
        example = labeler.create_cycle_example(ticker, df, c)
        
        assert example['ticker'] == ticker
        assert example['derived_features']['duration_days'] == 10
        assert len(example['context_window']) == 11 # 0 to 10 inclusive
        
        # Check file output
        files = os.listdir(mock_cycles_dir)
        assert f"{ticker}.jsonl" in files
        
        with open(os.path.join(mock_cycles_dir, f"{ticker}.jsonl"), 'r') as f:
            saved = json.loads(f.readline())
            assert saved['ticker'] == ticker
