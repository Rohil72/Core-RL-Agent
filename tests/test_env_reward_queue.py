import pytest
import numpy as np
import pandas as pd

pytestmark = pytest.mark.skip(
    reason="Tests archived CycleTradeEnv API (now TimeSteppedCycleEnv). "
           "Needs rewrite to match current env interface."
)

def create_mock_cycle(id, time_offset_days, confirm_label, confirm_delay_days=10):
    start = pd.Timestamp("2023-01-01", tz='UTC') + pd.Timedelta(days=time_offset_days)
    return {
        'cycle_id': id,
        'timestamp': start,
        'embedding': np.zeros(128, dtype=np.float32),
        'price_features': np.zeros(5, dtype=np.float32),
        'confirmation_label': confirm_label,
        'confirmation_time': start + pd.Timedelta(days=confirm_delay_days)
    }

class MockDataSource:
    def __init__(self, cycles):
        self.cycles = cycles
        self.idx = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.idx >= len(self.cycles):
            raise StopIteration
        c = self.cycles[self.idx]
        self.idx += 1
        return c
    def reset(self):
        self.idx = 0

def test_reward_queue_logic():
    # Scenario: 
    # Cycle 1: Flagged, Correct. Confirm at T+10.
    # Cycle 2: Flagged, Incorrect. Confirm at T+10. (Appears at T+5 relative to C1)
    # Cycle 3: Ignored (but confirmed). Confirm at T+10. (Appears at T+15 relative to C1)
    
    # Timeline:
    # T=0:  C1 arrives. Action: Flag.
    # T=5:  C2 arrives. Action: Flag. (C1 not yet confirmed)
    # T=10: C1 confirm time! (Should be paid when processing next step after T=10?)
    # T=15: C3 arrives. Action: Ignore. (C1 confirmed? C2 confirmed?)
    
    # Note: Env checks confirmation "up to current time" when moving to NEXT step.
    # If C2 arrives at T=5, we process C1->C2 transition. Time moves 0->5. 
    # C1 confirms at 10. So at T=5, C1 is NOT confirmed.
    # Next step: C2 -> C3 (T=15). Time moves 5->15.
    # C1 confirms at 10. 10 <= 15. So C1 should be paid.
    # C2 confirms at 5+10 = 15. 15 <= 15. So C2 should be paid too!
    
    cycles = [
        create_mock_cycle("c1", 0, True),
        create_mock_cycle("c2", 5, False),
        create_mock_cycle("c3", 15, True),
        create_mock_cycle("c4", 30, True) # Dummy to flush C3
    ]
    
    config = {
        'embedding_size': 128,
        'action_space_size': 3,
        'reward_params': {'c_flag': -0.1, 'R_confirm': 1.0, 'R_miss': -0.5}
    }
    
    env = CycleTradeEnv(config, data_source=MockDataSource(cycles))
    obs, info = env.reset()
    
    # Step 1: Process C1. Action Flag(1).
    # Returns immediate reward (-0.1)
    obs, reward, done, _, info = env.step(1)
    
    assert reward == -0.1
    assert env.episode_stats['flags'] == 1
    assert len(env.pending_confirmations) == 1
    
    # We are now at C2 (T=5). C1 pending.
    
    # Step 2: Process C2. Action Flag(1).
    # Immediate: -0.1.
    # Retroactive: 
    # Transition is C2(T=5) -> C3(T=15).
    # C1 confirms at T=10. 10 <= 15. Paid (+1.0).
    # C2 confirms at T=15. 15 <= 15. Paid (-0.5).
    # Total: -0.1 + 1.0 - 0.5 = 0.4
    obs, reward, done, _, info = env.step(1)
    
    # Float comparison with tolerance
    assert abs(reward - 0.4) < 1e-6
    assert len(env.pending_confirmations) == 0 # All flushed
    
    # We are now at C3 (T=15).
    # Cycle 1 confirm time = 10.
    # Cycle 2 confirm time = 15.
    # Current time update to 15? Yes, env._next_observation updates self.current_time.
    
    # Step 3: Process C3. Action Ignore(0).
    # Immediate: 0.
    # Retroactive: 
    # C1, C2 already paid.
    # C3 confirm is T=25. Next is C4 (T=30).
    # So C3 will be paid in NEXT step (if it was pending, but we ignored it and R_miss_ignore is 0).
    # So Total: 0.
    obs, reward, done, _, info = env.step(0)
    
    assert reward == 0.0
    assert len(env.pending_confirmations) == 0 # All flushed
    
    # We are now at C4 (T=30).
