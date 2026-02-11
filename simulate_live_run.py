import logging
import json
import time
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.trainers.train_rl import load_config
from src.trainers.online_loop import OnlineLoop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_live_run():
    config = load_config()
    
    # Setup Output
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("data/live_runs", exist_ok=True)
    log_file = f"data/live_runs/{run_id}.jsonl"
    
    logger.info(f"Starting simulation run: {run_id}")
    
    # Initialize Loop
    online_loop = OnlineLoop(config)
    
    # Mock Data Generator
    # Simulates ticks and confirmations
    ticker = "AAPL"
    start_time = pd.Timestamp.now(tz='UTC')
    
    with open(log_file, 'w') as f:
        for i in range(100):
            current_time = start_time + timedelta(minutes=i*15)
            
            # 1. Generate Price Tick
            price_row = {
                'open': 150 + np.random.randn(),
                'high': 155 + np.random.randn(),
                'low': 149 + np.random.randn(),
                'close': 152 + np.random.randn(),
                'volume': 1000000
            }
            
            # 2. Process Tick
            event = online_loop.process_tick(ticker, current_time, price_row)
            
            if event:
                logger.info(f"Event detected: {event}")
                f.write(json.dumps(event) + "\n")
                
                # simulate delayed confirmation for this event some steps later
                # We'll just confirm it immediately for the smoke test or use a queue
                # Let's say we retroactively confirm it at step i+5 (simulated by a separate check)
                
                # For simplicity here, we resolve it right away
                confirm_data = {'confirmed': bool(np.random.random() > 0.5)}
                reward_event = online_loop.process_confirmation(event['cycle_id'], confirm_data)
                
                if reward_event:
                    logger.info(f"Reward applied: {reward_event}")
                    f.write(json.dumps(reward_event) + "\n")
            
            # Sleep to simulate real-time speed if configured
            # time.sleep(0.01) 
            
    logger.info(f"Simulation finished. Log written to {log_file}")

if __name__ == "__main__":
    simulate_live_run()
