"""
Evaluation script to visualize agent predictions vs ground truth cycles.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
from src.envs.cycle_trade_env import TimeSteppedCycleEnv
from src.data.io_utils import read_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "models/rl_v2/final_model"
DATA_DIR = "data/precomputed"
OUTPUT_DIR = "data/evaluation_plots"

def plot_comparison(ticker, prices, ground_truth, predictions):
    """Generates side-by-side (top/bottom) charts of GT vs Agent."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # 1. Ground Truth
    ax1.plot(prices.index, prices.values, color='black', alpha=0.7, label='Price')
    ax1.set_title(f"{ticker} - Ground Truth (Detector)")
    
    # Shade Ground Truth
    # Find contiguous blocks of 1s in ground_truth (which is a list of 0/1)
    gt_array = np.array(ground_truth)
    diff = np.diff(np.concatenate([[0], gt_array, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    for s, e in zip(starts, ends):
        # Prevent index out of bounds
        e_idx = min(e, len(prices)-1)
        ax1.axvspan(prices.index[s], prices.index[e_idx], color='green', alpha=0.3)
    
    # 2. Agent Predictions
    ax2.plot(prices.index, prices.values, color='black', alpha=0.7, label='Price')
    ax2.set_title(f"{ticker} - Agent Predictions (PPO)")
    
    # Shade Agent Predictions
    pred_array = np.array(predictions)
    diff_pred = np.diff(np.concatenate([[0], pred_array, [0]]))
    starts_pred = np.where(diff_pred == 1)[0]
    ends_pred = np.where(diff_pred == -1)[0]
    
    for s, e in zip(starts_pred, ends_pred):
        e_idx = min(e, len(prices)-1)
        ax2.axvspan(prices.index[s], prices.index[e_idx], color='blue', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_comparison.png"))
    plt.close()

def evaluate():
    if not SB3_AVAILABLE:
        logger.error("stable-baselines3 is not installed. Cannot evaluate.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH + ".zip"):
        logger.error(f"Model not found at {MODEL_PATH}. Run training first.")
        return
        
    model = PPO.load(MODEL_PATH)
    files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    
    for f in files:
        ticker = os.path.basename(f).replace(".parquet", "")
        logger.info(f"Evaluating {ticker}...")
        
        df = read_dataframe(f).dropna()
        if df.empty: continue
        
        # We need the actual prices for plotting
        # Assuming OHLCV columns exist
        prices = df['close']
        
        env = TimeSteppedCycleEnv({}, df)
        obs, _ = env.reset()
        
        ground_truths = []
        predictions = []
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ground_truths.append(info['ground_truth'])
            predictions.append(action)
            done = terminated or truncated
            
        # Truncate prices to match ground_truths length if needed
        # (env might skip first window or last step)
        plot_prices = prices.iloc[:len(ground_truths)]
        
        plot_comparison(ticker, plot_prices, ground_truths, predictions)
        
        # Calculate metrics for the ticker
        gt_arr = np.array(ground_truths)
        pred_arr = np.array(predictions)
        accuracy = np.mean(gt_arr == pred_arr)
        logger.info(f"{ticker} Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate()
