"""
PPO training on TimeSteppedCycleEnv with real precomputed data.
"""

import os
import glob
import logging
import pandas as pd
import yaml
from typing import Dict, Any

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from src.envs.cycle_trade_env import TimeSteppedCycleEnv
from src.data.io_utils import read_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(path: str = "configs/rl.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_all_precomputed_data(pattern: str) -> pd.DataFrame:
    """Loads and concatenates all precomputed parquet files."""
    files = glob.glob(pattern)
    if not files:
        logger.warning(f"No precomputed files found matching: {pattern}")
        return pd.DataFrame()
    
    all_dfs = []
    for f in files:
        logger.info(f"Loading {f}...")
        all_dfs.append(read_dataframe(f))
        
    df = pd.concat(all_dfs, ignore_index=True)
    return df.dropna()

def train(config_path: str = "configs/rl.yaml"):
    config = load_config(config_path)

    if not SB3_AVAILABLE:
        logger.error("stable-baselines3 is not installed.")
        return

    # 1. Load Real Data
    data_pattern = config['data']['precomputed_dir']
    full_df = load_all_precomputed_data(data_pattern)
    
    if full_df.empty:
        logger.error("No training data available. Run scripts/precompute_ground_truth.py first.")
        return
    
    logger.info(f"Loaded {len(full_df)} rows of training data.")

    # 2. Setup Environment
    env_cfg = config["env"]
    train_cfg = config["training"]

    # We use DummyVecEnv for stable training
    def make_env():
        return TimeSteppedCycleEnv(env_cfg, full_df)
        
    env = DummyVecEnv([make_env])

    # 3. Setup Model
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        n_steps=train_cfg.get("n_steps", 2048),
        batch_size=train_cfg.get("batch_size", 128),
        gamma=train_cfg.get("gamma", 0.99),
        verbose=1,
        tensorboard_log="logs/ppo_v2"
    )

    # 4. Train
    logger.info("Starting PPO training on real data...")
    model.learn(total_timesteps=train_cfg["total_timesteps"])

    # 5. Save
    save_dir = train_cfg.get("model_dir", "models/rl_v2")
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, "final_model"))
    logger.info(f"Model saved to {save_dir}/final_model")

if __name__ == "__main__":
    train()
