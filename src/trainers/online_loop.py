import logging
import time
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Deque
from collections import deque
from datetime import datetime

# Integration with existing components
from src.data.streaming_api import CycleLabelStreamer
# Assuming we have a way to load the model
try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

logger = logging.getLogger(__name__)

class ReplayBuffer:
    """
    Simple experience replay buffer for online fine-tuning.
    Stores tuples of (observation, action, reward, next_observation, done).
    But here we have DELAYED rewards.
    So we store (episode_id, observation, action) and wait for reward to backfill.
    """
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        # Staging area for episodes waiting for confirmation
        self.pending_episodes: Dict[str, Dict] = {} 
        # Completed experiences ready for training: (obs, action, reward, next_obs, done)
        self.buffer: Deque = deque(maxlen=capacity)
        
    def add_step(self, cycle_id: str, obs: Dict, action: int):
        """
        Record a step taken by the agent.
        """
        self.pending_episodes[cycle_id] = {
            'obs': obs,
            'action': action,
            'timestamp': pd.Timestamp.now() # Real time or sim time?
        }
        
    def complete_episode(self, cycle_id: str, reward: float, next_obs: Dict, done: bool):
        """
        Called when confirmation arrives and we can compute the final reward.
        """
        if cycle_id in self.pending_episodes:
            step = self.pending_episodes.pop(cycle_id)
            experience = (step['obs'], step['action'], reward, next_obs, done)
            self.buffer.append(experience)
            
    def sample(self, batch_size: int):
        import random
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)

class OnlineLoop:
    def __init__(self, config: Dict[str, Any], model_path: str = None):
        self.config = config
        self.model = self._load_model(model_path)
        self.replay_buffer = ReplayBuffer(capacity=config['online'].get('replay_buffer_size', 10000))
        self.streamer = CycleLabelStreamer(ticker="DUMMY") # Ticker updated per usage?
        
        # State
        self.running = False
        
    def _load_model(self, path):
        if PPO and path and os.path.exists(path):
            return PPO.load(path)
        logger.warning("No model loaded or SB3 not available. Using random policy.")
        return None
        
    def process_tick(self, ticker: str, timestamp: pd.Timestamp, price_row: dict) -> Optional[Dict]:
        """
        Ingest a tick, potentially detecting a cycle.
        If cycle detected -> Run inference -> Store action.
        """
        # Note: CycleLabelStreamer is per ticker. In a real loop we'd manage multiple streamers.
        # For simplicity, we assume this loop handles one ticker or we re-instantiate/map streamers.
        # Let's assume we have a dict of streamers if multiple tickers.
        
        # For this implementation, we just use the single streamer for the passed ticker (re-configured or separate logic).
        # Actually CycleLabelStreamer keeps state (history). So we need one per ticker.
        # Let's just assume single ticker usage for the 'simulate_live_run' demo.
        if self.streamer.ticker != ticker:
             self.streamer = CycleLabelStreamer(ticker=ticker) # Reset if ticker changes (basic logic)
             
        cycle_example = self.streamer.process_new_tick(timestamp, price_row)
        
        result_event = None
        
        if cycle_example:
            # 1. Transform to Observation
            # Env observation: embedding, price_features
            # cycle_example probably doesn't have embedding yet unless data pipeline runs it.
            # We need an Encoder! 
            # "Each detected cycle is encoded using the pre-trained encoder checkpoint"
            # We assume a function `encode_cycle(cycle_example)` exists or mock it.
            obs = self._encode_cycle(cycle_example)
            
            # 2. Inference
            if self.model:
                action, _ = self.model.predict(obs, deterministic=True)
                # If obs is dict, SB3 handles it.
            else:
                action = np.random.randint(0, 3)
                
            # 3. Store in Buffer (Pending)
            cycle_id = f"{ticker}_{cycle_example['cycle_end']}" # Unique ID
            self.replay_buffer.add_step(cycle_id, obs, action)
            
            result_event = {
                "type": "ACTION",
                "ticker": ticker,
                "cycle_id": cycle_id,
                "action": int(action),
                "timestamp": timestamp.isoformat()
            }
            
            # 4. Check for Confirmation (Immediate or delayed?)
            # The streamer might not have future data. 
            # In simulation, we might feed "confirmation" events separately.
            
        return result_event

    def process_confirmation(self, cycle_id: str, confirmation_data: Dict):
        """
        External system tells us a cycle is confirmed/rejected.
        """
        # Calculate Reward
        # We need the action. It's in pending_episodes.
        if cycle_id in self.replay_buffer.pending_episodes:
            step = self.replay_buffer.pending_episodes[cycle_id]
            action = step['action']
            
            reward = 0.0
            params = self.config['env']['reward_params']
            
            # Immediate cost (should have been applied? In RL env it is step reward. Here we bundle.)
            if action != 0:
                reward += params['c_flag']
                
            confirmed = confirmation_data.get('confirmed', False)
            
            if action != 0:
                if confirmed:
                    reward += params['R_confirm']
                else:
                    reward += params['R_miss']
            elif params.get('R_miss_ignore', 0.0) != 0.0:
                 if confirmed:
                     reward += params['R_miss_ignore']
                     
            # Close episode
            # Next obs? For single step episodes, next obs is terminal (dummy).
            next_obs = step['obs'] # Or zeros
            done = True
            
            self.replay_buffer.complete_episode(cycle_id, reward, next_obs, done)
            
            # Fine-tune?
            self._finetune_step()
            
            return {
                "type": "REWARD",
                "cycle_id": cycle_id,
                "reward": reward,
                "confirmed": confirmed
            }
        return None

    def _encode_cycle(self, cycle_example: Dict) -> Dict:
        """
        Mock encoder or call real model.
        """
        # Mock embedding
        emb = np.random.randn(self.config['env']['embedding_size']).astype(np.float32)
        feats = np.random.randn(5).astype(np.float32)
        
        return {
            "embedding": emb,
            "price_features": feats
        }

    def _finetune_step(self):
        """
        Trigger a gradient update if enough data.
        """
        if len(self.replay_buffer.buffer) >= self.config['training']['batch_size']:
            # Fine-tune logic
            # Extract batch and train
            pass
