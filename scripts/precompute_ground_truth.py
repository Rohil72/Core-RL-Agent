"""
Precompute ground truth cycles and features for RL training.
"""

import os
import logging
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from src.data.loader import load_market_data
from src.data.features import compute_technical_features, compute_fundamental_features_aligned
from src.cycle.cycle_detector import detect_cycles
from src.models.timesnet_encoder import TimesNetEncoder
from src.data.io_utils import write_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "data/precomputed"
CONFIG_PATH = "config/market_universe.yaml"
START = "2018-01-01"
END = "2024-12-31"

def precompute():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    data = load_market_data(CONFIG_PATH, START, END)
    
    # 2. Init Encoder
    encoder = TimesNetEncoder(in_dim=5, embed_dim=128).to(DEVICE)
    encoder.eval()
    
    all_processed = []

    for ticker, dfs in tqdm(data.items(), desc="Processing tickers"):
        price_df = dfs['price']
        earnings_df = dfs['earnings']
        
        if price_df.empty: continue
        
        # 3. Compute Features
        price_df = compute_technical_features(price_df)
        if not earnings_df.empty:
            price_df = compute_fundamental_features_aligned(price_df, earnings_df)
        else:
            # Add dummy fundamental cols if missing
            for col in ['fund_eps_surprise', 'fund_eps_growth_yoy', 'fund_eps_accel']:
                price_df[col] = 0.0
        
        # 4. Detect Cycles (Ground Truth)
        cycles = detect_cycles(price_df['close'])
        
        # Create In-Cycle Mask
        price_df['in_cycle'] = 0
        for c in cycles:
            mask = (price_df.index >= c.start_date) & (price_df.index <= c.end_date)
            price_df.loc[mask, 'in_cycle'] = 1
            
        # 5. Generate TimesNet Embeddings (Sliding Window)
        # We need a window of OHLCV to feed TimesNet. 
        # For simplicity, we'll do this once per day (trailing 96 days as often used in TimesNet)
        window_size = 96 
        embeddings = []
        
        # Pre-pad price data for windowing
        padded_prices = price_df[['open', 'high', 'low', 'close', 'volume']].values
        
        # We only compute embeddings where we have a full window
        indices = []
        for i in range(window_size, len(price_df)):
            window = padded_prices[i-window_size:i]
            # Normalize window
            window_mean = window.mean(axis=0)
            window_std = window.std(axis=0) + 1e-6
            window_norm = (window - window_mean) / window_std
            
            with torch.no_grad():
                inp = torch.FloatTensor(window_norm).unsqueeze(0).to(DEVICE) # (1, W, 5)
                emb = encoder(inp) # (1, 1, 128) or (1, 128) based on impl
                # Adjust depending on TimesNetEncoder output shape
                if len(emb.shape) == 3:
                    emb = emb.mean(dim=1)
                embeddings.append(emb.cpu().numpy().flatten())
                indices.append(price_df.index[i])
        
        # Create Embedding DataFrame
        emb_df = pd.DataFrame(embeddings, index=indices)
        emb_df.columns = [f'emb_{i}' for i in range(emb_df.shape[1])]
        
        # Merge back
        final_df = price_df.join(emb_df, how='inner')
        final_df['ticker'] = ticker
        
        # Save per ticker
        write_dataframe(final_df, os.path.join(OUTPUT_DIR, f"{ticker}.parquet"))
        all_processed.append(final_df)

    logger.info(f"Precomputation complete. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    precompute()
