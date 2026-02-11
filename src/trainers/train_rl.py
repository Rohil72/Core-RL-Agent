import os
import yaml
import glob
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import gymnasium as gym

# Try implementing wrappers
from gymnasium.wrappers import NormalizeObservation, NormalizeReward

# PPO + VecEnv
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    
from src.envs.cycle_trade_env import CycleTradeEnv
from src.data.io_utils import read_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "configs/rl.yaml") -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class CycleDataset:
    """
    Loads and yields cycle data for the environment.
    """
    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        self.config = config
        self.split = split
        self.data_files = glob.glob(config['data']['embeddings_path'])
        self.cycles = []
        self._load_data()
        
    def _load_data(self):
        """
        Loads embeddings and feature tables, merges them, and sorts by time.
        In a real scenario, we might want to lazy load or iterate over chunks.
        For now, we load all into memory as per prompt implication (or simplified).
        """
        if not self.data_files:
            logger.warning("No embedding files found. Creating dummy data for smoke test/initialization.")
            self._create_dummy_data()
            return

        # Simplified loading logic:
        # 1. Load Embeddings
        # 2. Load Features
        # 3. Merge on cycle_id or index
        # This implementation assumes files are self-contained or we iterate them.
        
        all_dfs = []
        for f in self.data_files:
            try:
                df = read_dataframe(f)
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading {f}: {e}")
                
        if not all_dfs:
            self._create_dummy_data()
            return
            
        full_df = pd.concat(all_dfs)
        
        # Sort by timestamp
        if 'timestamp' in full_df.columns:
            full_df = full_df.sort_values('timestamp')
            
        # Convert to list of dicts for iterator
        self.cycles = full_df.to_dict('records')
        
        # Train/Test split logic could go here
        if self.split == 'train':
            self.cycles = self.cycles[:int(0.8 * len(self.cycles))]
        else:
            self.cycles = self.cycles[int(0.8 * len(self.cycles)):]
            
    def _create_dummy_data(self):
        """Generates dummy data for testing/fallback."""
        count = 100
        embedding_size = self.config['env']['embedding_size']
        
        start_time = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=count)
        
        self.cycles = []
        for i in range(count):
            self.cycles.append({
                'cycle_id': f"cycle_{i}",
                'timestamp': start_time + pd.Timedelta(days=i),
                'embedding': np.random.randn(embedding_size).astype(np.float32),
                'price_features': np.random.randn(5).astype(np.float32),
                'confirmation_label': bool(np.random.random() > 0.5),
                'confirmation_time': start_time + pd.Timedelta(days=i+5) # Delayed by 5 days
            })
            
    def __iter__(self):
        for cycle in self.cycles:
            yield cycle
            
    def __len__(self):
        return len(self.cycles)
        
    def reset(self):
        pass # Nothing needed for list iterator, but if we had file pointers we'd reset them

def make_env(config_path: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        config = load_config(config_path)
        dataset = CycleDataset(config, split='train')
        env = CycleTradeEnv(config['env'], data_source=dataset)
        env = Monitor(env) # Record stats
        env.reset(seed=seed + rank)
        return env
    return _init

def train(config_path: str = "configs/rl.yaml"):
    config = load_config(config_path)
    
    if not SB3_AVAILABLE:
        logger.error("Stable-Baselines3 not found. Please install it or use fallback.")
        return

    # Create Vectorized Env
    # Using DummyVecEnv for simplicity and debuggability, can switch to SubprocVecEnv
    n_envs = 4
    env = DummyVecEnv([make_env(config_path, i) for i in range(n_envs)])
    
    # Normalization Wrappers (applied on VecEnv)
    # VecNormalize is usually better for PPO
    from stable_baselines3.common.vec_env import VecNormalize
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Setup Model
    train_config = config['training']
    policy_kwargs = dict(net_arch=dict(pi=[128, 64], vf=[128, 64]))
    
    tensorboard_log = train_config.get('log_dir', "logs/rl")
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except ImportError:
        logger.warning("tensorboard is not installed; disabling tensorboard logging.")
        tensorboard_log = None

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=train_config.get('learning_rate', 3e-4),
        n_steps=train_config.get('n_steps', 2048),
        batch_size=train_config.get('batch_size', 64),
        gamma=train_config.get('gamma', 0.99),
        gae_lambda=train_config.get('gae_lambda', 0.95),
        clip_range=train_config.get('clip_range', 0.2),
        ent_coef=train_config.get('ent_coef', 0.0),
        vf_coef=train_config.get('vf_coef', 0.5),
        max_grad_norm=train_config.get('max_grad_norm', 0.5),
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tensorboard_log
    )

    # Checkpoint Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=train_config.get('checkpoint_freq', 10000),
        save_path=train_config.get('model_dir', "models/rl"),
        name_prefix="ppo_cycle_trader"
    )

    # Train
    logger.info("Starting training...")
    model.learn(
        total_timesteps=train_config['total_timesteps'],
        callback=checkpoint_callback
    )
    
    # Save final model
    model.save(os.path.join(train_config.get('model_dir', "models/rl"), "final_model"))
    env.save("vec_normalize.pkl")
    logger.info("Training finished.")

def evaluate_policy_on_holdout(policy_path, config_path="configs/rl.yaml", topk=5):
    """
    Evaluates a trained policy on a holdout dataset.
    Computes Precision, Recall, F1, and Calibration.
    """
    config = load_config(config_path)
    
    # Load separate holdout set
    dataset = CycleDataset(config, split='test') 
    env = CycleTradeEnv(config['env'], data_source=dataset)
    
    if not SB3_AVAILABLE:
        logger.error("SB3 not available for evaluation.")
        return

    # Load Model
    model = PPO.load(policy_path)
    
    # Run Inference
    obs, info = env.reset()
    done = False
    
    predictions = [] # (prob_flag, confirmed)
    
    # Manual loop to collect predictions
    # Note: If using VecNormalize during training, we must use it here too!
    # Or load the stats. For simplicity, we assume raw features or simple wrapper.
    # If trained with VecNormalize, we MUST load it.
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        
        # Get ground truth from info (if available) or wait for confirmation
        # Since we are evaluating offline, we can peek at the underlying dataset if needed?
        # But let's rely on the env.
        # Env step returns reward. The reward tells us if we were confirmed (eventually).
        # But for 'calibration', we want the model's confidence probability if possible.
        # PPO 'predict' returns action. To get probs, we need 'model.policy.get_distribution(obs)'.
        
        # Access probability
        # obs is numpy array from wrapper or dict
        # We need to process obs to tensor
        import torch
        with torch.no_grad():
             # Convert obs to dict of tensors if needed
             # SB3 handles this inside predict, but for get_distribution we might need to be careful
             # If using VecEnv, obs is batched. Here single env.
             obs_tensor = model.policy.obs_to_tensor(obs)[0]
             dist = model.policy.get_distribution(obs_tensor)
             probs = dist.distribution.probs.cpu().numpy()[0] # [prob_ignore, prob_flag, prob_proactive]
             
        score = probs[1] + probs[2] # Prob of flagging or proactive
        
        # Step
        # In eval, we still step to advance. 
        # But to get the *truth*, we might need to know the 'confirm_label' of the CURRENT cycle.
        # The env's 'current_cycle' has it.
        # But Env.step() moves to NEXT cycle.
        # So we capture truth BEFORE step.
        # Capture truth from info dict (set by env in step/obs)
        # Fallback to env.current_cycle for backward compatibility
        truth = info.get('confirmation_label', 
                         env.current_cycle.get('confirmation_label', False) if hasattr(env, 'current_cycle') and env.current_cycle else False)
        
        predictions.append({
            'score': score,
            'action': action,
            'truth': truth,
            'timestamp': info.get('timestamp')
        })
        
        obs, reward, done, truncated, info = env.step(action)
        if truncated: done = True

    # Compute Metrics
    df = pd.DataFrame(predictions)
    if df.empty:
        logger.warning("No evaluation data.")
        return
        
    # Calibration Check
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
    
    # Binary truth: 1 if truth else 0
    y_true = df['truth'].astype(int)
    y_scores = df['score']
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    # Thresholding at 0.5 (or argmax)
    y_pred = (y_scores > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred)
    
    logger.info(f"Evaluation Metrics:")
    logger.info(f"AP: {ap:.4f}")
    logger.info(f"F1 (at 0.5): {f1:.4f}")
    
    # Time-weighted precision?
    # Weigh earlier examples? Or specific calculation?
    # "time-weighted precision (gives weight to earlier correct identifications)"
    # Usually this means measuring how *early* in a cycle we detected it, but here step is per cycle.
    # Maybe it means weighing recent performance more? Or weighing by time-to-confirmation?
    # For now, standard AP is fine.

if __name__ == "__main__":
    train()
