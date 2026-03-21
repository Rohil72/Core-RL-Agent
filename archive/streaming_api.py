import pandas as pd
import numpy as np
import os
import logging
from typing import Optional, List, Any
from src.cycle.cycle_detector import detect_cycles, Cycle
from src.data.labeler import create_cycle_example
from src.data.fetcher import fetch_price_history

logger = logging.getLogger(__name__)

class CycleLabelStreamer:
    """
    Streaming processor that receives ticks, detects cycles, and labels them.
    """
    def __init__(self, ticker: str, detector=None, fetcher=None, labeler=None, out_path: str = None):
        """
        Args:
            ticker: The ticker symbol.
            detector: (Optional) The detection function or object. Defaults to src.cycle.cycle_detector.detect_cycles.
            fetcher: (Optional) Unused in direct stream, but kept for interface compatibility if needed.
            labeler: (Optional) Unused in direct stream, utilizing src.data.labeler.
            out_path: (Optional) Directory or path for output. Defaults to standard data/cycles location handled by labeler.
        """
        self.ticker = ticker
        self.detector = detector if detector else detect_cycles
        # Buffer for price history
        self.history_df = pd.DataFrame()
        # Keep track of processed cycles (by start_date, end_date) to avoid duplicates
        self.processed_cycles = set()
        
    def process_new_tick(self, timestamp: pd.Timestamp, price_row: dict) -> Optional[dict]:
        """
        Ingest one new row (tick/day), update buffer, detect cycles, and label if new cycle found.
        """
        # 1. Update Buffer
        # Create a DataFrame for the new row
        row_df = pd.DataFrame([price_row], index=[timestamp])
        
        # Ensure UTC
        if row_df.index.tz is None:
            row_df.index = row_df.index.tz_localize('UTC')
        else:
            row_df.index = row_df.index.tz_convert('UTC')
            
        # Append
        if self.history_df.empty:
            self.history_df = row_df
        else:
            # Avoid duplicates if timestamp already exists
            if timestamp in self.history_df.index:
                self.history_df.loc[timestamp] = row_df.iloc[0]
            else:
                self.history_df = pd.concat([self.history_df, row_df])
                
        self.history_df.sort_index(inplace=True)
        
        # 2. Detect Cycles
        # We need enough data.
        if len(self.history_df) < 40: # Min duration in detector default
            return None
            
        cycles = self.detector(self.history_df['close'])
        
        # 3. Process Newly Closed Cycles
        # Identifying "new" cycles. 
        # A cycle is "new" if we haven't processed it yet.
        # Ideally, we only care about cycles that *end* at the current timestamp, 
        # because those are the ones that just finished.
        
        new_example = None
        
        for cycle in cycles:
            # Check if processed
            # cycle is a dataclass.
            c_id = (cycle.start_date, cycle.end_date)
            
            if c_id in self.processed_cycles:
                continue
                
            # Is this cycle confirmed/finished? 
            # The detector returns it, so it is a candidate.
            # If the cycle end date is strictly in the past, we might have missed it or it's historical.
            # If the cycle end date is the current timestamp (or close to it), it's a "just closed" cycle.
            
            # Since we want to label it, we might need future returns (labeler logic).
            # The labeler computes returns for [21, 63, 252] horizons.
            # If we are live streaming, we CANNOT compute future returns yet.
            # We can only compute the "detected" part. 
            # The prompt says: "convert cycles into labeled examples ... output ... to receive new ticks/cycles in real time."
            # "compute forward realized returns ... If price lookup out of bounds, set np.nan"
            # So we SHOULD produce the example even if future returns are NaN.
            
            # We process it.
            try:
                example = create_cycle_example(self.ticker, self.history_df, cycle)
                if example:
                    self.processed_cycles.add(c_id)
                    new_example = example
            except Exception as e:
                logger.error(f"Error creating cycle example for {self.ticker}: {e}")
                
        return new_example

