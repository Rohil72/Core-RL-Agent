import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Deque
from collections import deque
import logging

logger = logging.getLogger(__name__)

class CycleTradeEnv(gym.Env):
    """
    Gym environment for cycle detection and trading.
    Agents observe cycle embeddings and features, and decide whether to flag them as genuine.
    Rewards are delayed until fundamental confirmation is available.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config: Dict[str, Any], data_source=None):
        """
        Args:
            config: Dictionary containing environment configuration.
            data_source: Iterator or generator that yields cycle data (observation, confirmation_info).
                         In offline mode, this yields historical cycles.
                         In online mode, this might yield live cycles.
        """
        super(CycleTradeEnv, self).__init__()
        self.config = config
        self.data_source = data_source
        self.cycle_iterator = None

        # Configuration Params
        self.embedding_size = config.get('embedding_size', 128)
        self.action_space_size = config.get('action_space_size', 3)
        self.reward_params = config.get('reward_params', {
            'c_flag': -0.01,
            'R_confirm': 1.0,
            'R_miss': -0.5,
            'R_miss_ignore': -0.0
        })
        
        # Action Space: 0: Ignore, 1: Flag, 2: Proactive
        self.action_space = spaces.Discrete(self.action_space_size)

        # Observation Space
        # We use a Dict space for flexibility
        self.observation_space = spaces.Dict({
            "embedding": spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_size,), dtype=np.float32),
            "price_features": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32), # Example size
            # Cycle meta could be encoded or passed as raw features. 
            # For simplicity in valid Gym spaces, we keep numeric parts here.
            # "cycle_meta": spaces.Box(...) 
        })
        
        # State
        self.current_cycle = None
        self.done = False
        
        # Delayed Reward Queue
        # Stores tuples: (cycle_id, action, due_timestamp, expected_reward_function)
        # or simplified: (cycle_id, action, confirmation_data)
        self.pending_confirmations: Deque[Dict] = deque()
        self.current_time = None # simulation time

        # Metrics
        self.episode_stats = {
            'flags': 0,
            'confirmed_flags': 0,
            'false_flags': 0,
            'total_reward': 0.0
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.data_source is None:
            # If no data source provided, raise error
            raise ValueError("No data_source provided to environment. Cannot reset without data.")
        elif hasattr(self.data_source, 'reset'):
             self.data_source.reset()
             self.cycle_iterator = iter(self.data_source)
        else:
             # Assume it's iterable
             self.cycle_iterator = iter(self.data_source)
             
        self.pending_confirmations.clear()
        self.episode_stats = {k: 0 for k in self.episode_stats}
        self.current_time = pd.Timestamp.min.tz_localize('UTC') # Start time with UTC timezone
        
        return self._next_observation()

    def _next_observation(self):
        """
        Advances to the next cycle in the stream.
        Handles checking the pending queue for any resolutions that can be applied NOW
        (if we are in off-line mode where we jump time, but usually we resolve *before* step returns).
        """
        try:
            self.current_cycle = next(self.cycle_iterator)
        except StopIteration:
            self.done = True
            # Return dummy observation
            dummy_obs = {
                "embedding": np.zeros(self.embedding_size, dtype=np.float32),
                "price_features": np.zeros(5, dtype=np.float32)
            }
            return dummy_obs, {}
            
        # Parse cycle data
        # Assume current_cycle is a dict with:
        # 'embedding', 'features', 'meta', 'timestamp', 'confirmation_label', 'confirmation_time'
        
        obs = {
            "embedding": self.current_cycle.get('embedding', np.zeros(self.embedding_size, dtype=np.float32)),
            "price_features": self.current_cycle.get('price_features', np.zeros(5, dtype=np.float32))
        }
        
        # Update simulation time to current cycle time
        if 'timestamp' in self.current_cycle:
             cycle_time = pd.to_datetime(self.current_cycle['timestamp'])
             # Ensure UTC timezone for comparison
             if cycle_time.tz is None:
                 cycle_time = cycle_time.tz_localize('UTC')
             else:
                 cycle_time = cycle_time.tz_convert('UTC')
             # Ensure monotonic time for safety, though data should be sorted
             if cycle_time > self.current_time:
                 self.current_time = cycle_time
                 
        info = {
            "cycle_id": self.current_cycle.get('cycle_id'),
            "timestamp": self.current_cycle.get('timestamp'),
            "confirmation_label": self.current_cycle.get('confirmation_label', False),
        }
        
        return obs, info

    def step(self, action):
        if self.done:
            obs, info = self._next_observation()
            return obs, 0.0, True, False, info

        current_reward = 0.0
        
        # 1. Immediate Reward / Cost
        if action == 1 or action == 2: # Flag or Proactive
            current_reward += self.reward_params['c_flag']
            self.episode_stats['flags'] += 1
            
        # 2. Queue for Delayed Reward
        # We need the confirmation logic. 
        # In this env, we assume the labeled data *already knows* the outcome.
        # But to simulate the *delay*, we might check if 'confirmation_time' is <= current_time.
        # However, for strictly sequential offline training, 'current_time' jumps from cycle to cycle.
        # Any confirmation that happened *between* the previous cycle and this one should be paid out.
        # OR: We can pay it out *immediately* if we are training offline and want to simplify credit assignment without a delay buffer (Monte Carlo style).
        # BUT: The prompt asks for a mechanism to queue outstanding cycles.
        
        # Let's add this cycle to pending
        if action != 0:
             confirm_time = pd.to_datetime(self.current_cycle.get('confirmation_time', pd.Timestamp.max))
             # Ensure UTC timezone
             if confirm_time.tz is None:
                 confirm_time = confirm_time.tz_localize('UTC')
             else:
                 confirm_time = confirm_time.tz_convert('UTC')
                 
             self.pending_confirmations.append({
                 'cycle_id': self.current_cycle.get('cycle_id'),
                 'action': action,
                 'confirm_label': self.current_cycle.get('confirmation_label', False), # True/False
                 'confirm_time': confirm_time,
                 'params': self.reward_params
             })
        elif self.reward_params.get('R_miss_ignore', 0.0) != 0.0:
             # Also track ignored ones if we want to punish missed opportunities
             confirm_time = pd.to_datetime(self.current_cycle.get('confirmation_time', pd.Timestamp.max))
             # Ensure UTC timezone
             if confirm_time.tz is None:
                 confirm_time = confirm_time.tz_localize('UTC')
             else:
                 confirm_time = confirm_time.tz_convert('UTC')
                 
             self.pending_confirmations.append({
                 'cycle_id': self.current_cycle.get('cycle_id'),
                 'action': action, # 0
                 'confirm_label': self.current_cycle.get('confirmation_label', False),
                 'confirm_time': confirm_time,
                 'params': self.reward_params
             })

        # 3. Process Pending Rewards (Retroactive)
        # We check if any pending confirmations have 'occurred' by the time of the *next* cycle.
        # NOTE: This depends on the next cycle's timestamp.
        # We need to peek at next cycle or do this check at the start of next step?
        # Standard Gym: Step transitions state (S, A) -> (S', R). 
        # S' is the next cycle. So we can check strict time ordering:
        # pending items where confirm_time <= S'.timestamp are resolved.
        
        # Get next observation
        obs_next, info_next = self._next_observation()
        
        # If done, resolve ALL pending (end of episode/historical data)
        resolve_until = pd.Timestamp.max.tz_localize('UTC') if self.done else self.current_time
        
        resolved_reward = self._resolve_pending_rewards(resolve_until)
        total_reward = current_reward + resolved_reward
        
        self.episode_stats['total_reward'] += total_reward
        
        # Info updates
        info_next['resolved_reward'] = resolved_reward
        info_next['stats'] = self.episode_stats.copy()

        return obs_next, total_reward, self.done, False, info_next
        
    def _resolve_pending_rewards(self, time_limit):
        """
        Check pending confirmations that have resolved by `time_limit`.
        """
        reward = 0.0
        remaining = deque()
        
        while self.pending_confirmations:
            item = self.pending_confirmations.popleft()
            
            # Check if confirmation time is passed
            if item['confirm_time'] <= time_limit:
                # Resolved
                r = 0.0
                action = item['action']
                confirmed = item['confirm_label']
                params = item['params']
                
                if action > 0: # Flagged
                    if confirmed:
                        r = params['R_confirm']
                        self.episode_stats['confirmed_flags'] += 1
                    else:
                        r = params['R_miss']
                        self.episode_stats['false_flags'] += 1
                else: # Ignored
                    if confirmed and params.get('R_miss_ignore', 0.0) != 0.0:
                        r = params['R_miss_ignore']
                        
                reward += r
            else:
                # Not yet resolved, keep it
                remaining.append(item)
        
        # Restore remaining (preserve order)
        # Since we popped from left (oldest), and we stop? 
        # Actually, confirmation times might not be sorted if cycles overlap.
        # So we iterate all.
        # But 'remaining' contains those NOT resolved.
        # The while loop above drained everything. 
        # We need to put back unresolved ones.
        self.pending_confirmations = remaining
        
        return reward

    def render(self, mode='human'):
        print(f"Current Cycle: {self.current_cycle.get('cycle_id')} | Time: {self.current_time}")
        print(f"Stats: {self.episode_stats}")
