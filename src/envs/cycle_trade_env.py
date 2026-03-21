"""
Time-stepped Gym environment for cycle trading research.
Agents observe daily embeddings and features, predicting if "In Cycle".
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

class TimeSteppedCycleEnv(gym.Env):
    """
    Environment where each step is one trading day.
    Observation: TimesNet Embedding + Tech Features + Fundamental Features.
    Action: 0 = Out of Cycle, 1 = In Cycle.
    Reward: Accuracy vs Ground Truth (Cycle Detector).
    """
    
    def __init__(self, config: Dict[str, Any], data: pd.DataFrame):
        super().__init__()
        self.config = config
        self.df = data.reset_index(drop=True)
        self.current_step = 0
        
        # Identify Columns
        self.emb_cols = [c for c in self.df.columns if c.startswith('emb_')]
        self.tech_cols = [c for c in self.df.columns if c.startswith('tech_')]
        self.fund_cols = [c for c in self.df.columns if c.startswith('fund_')]
        
        self.embedding_size = len(self.emb_cols)
        self.tech_size = len(self.tech_cols)
        self.fund_size = len(self.fund_cols)
        
        # Action Space: 0 = Out, 1 = In
        self.action_space = spaces.Discrete(2)
        
        # Observation Space
        self.observation_space = spaces.Dict({
            "embedding": spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_size,), dtype=np.float32),
            "tech_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.tech_size,), dtype=np.float32),
            "fund_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.fund_size,), dtype=np.float32)
        })
        
        # Reward Params
        self.reward_params = config.get('reward_params', {
            'correct_in': 1.0,      # Correctly predicted 'In Cycle'
            'correct_out': 0.1,     # Correctly predicted 'Out of Cycle'
            'false_pos': -0.5,      # Flagged 'In' when 'Out'
            'false_neg': -1.0,      # Flagged 'Out' when 'In'
            'duration_bonus': 0.01  # Bonus per day for sustained correct In-Cycle
        })
        
        self.consecutive_correct_in = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.consecutive_correct_in = 0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        return {
            "embedding": row[self.emb_cols].values.astype(np.float32),
            "tech_features": row[self.tech_cols].values.astype(np.float32),
            "fund_features": row[self.fund_cols].values.astype(np.float32)
        }

    def step(self, action):
        row = self.df.iloc[self.current_step]
        ground_truth = int(row['in_cycle'])
        
        reward = 0.0
        if action == 1: # Predict In-Cycle
            if ground_truth == 1:
                reward = self.reward_params['correct_in']
                self.consecutive_correct_in += 1
                reward += self.consecutive_correct_in * self.reward_params['duration_bonus']
            else:
                reward = self.reward_params['false_pos']
                self.consecutive_correct_in = 0
        else: # Predict Out-of-Cycle
            if ground_truth == 0:
                reward = self.reward_params['correct_out']
            else:
                reward = self.reward_params['false_neg']
            self.consecutive_correct_in = 0
            
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        if done:
            # Clamp to last valid index to avoid IndexError
            self.current_step = min(self.current_step, len(self.df) - 1)
        obs = self._get_obs()
        
        info = {
            "ground_truth": ground_truth,
            "step": self.current_step,
            "ticker": self.df.iloc[0]['ticker'] if 'ticker' in self.df.columns else "unknown"
        }
        
        return obs, reward, done, False, info
