import time
import os
import re

LOG_PATH = r"C:\Users\rohil\.gemini\antigravity-ide\brain\f3e11d51-b086-42c5-a7dc-84a214449c34\.system_generated\tasks\task-190.log"

def monitor_log():
    if not os.path.exists(LOG_PATH):
        print(f"Waiting for log file {LOG_PATH} to be created...")
        while not os.path.exists(LOG_PATH):
            time.sleep(1)
            
    print("Monitoring Phase 4 Sweep Progress... (Press Ctrl+C to stop)\n")
    
    current_variant = "Unknown"
    current_seed = "Unknown"
    current_epoch = 1
    current_batch = 0
    current_loss = 0.0
    
    with open(LOG_PATH, 'r') as f:
        # Seek to the end or read from beginning depending on what user wants. 
        # Better to read from beginning to catch the current state quickly.
        while True:
            line = f.readline()
            if not line:
                # Format and print current status
                status = f"\r[Live] Variant: {current_variant: <25} | Seed: {current_seed: <2} | Epoch: {current_epoch: <3} | Batch: {current_batch: <4} | Loss: {current_loss:.5f}      "
                print(status, end="", flush=True)
                time.sleep(0.5)
                continue
                
            line = line.strip()
            
            # Match variant and seed
            # === Starting Run: Variant huber_baseline | Seed 7 ===
            if "=== Starting Run: Variant" in line:
                match = re.search(r"Variant\s+([^\s]+)\s+\|\s+Seed\s+(\d+)", line)
                if match:
                    current_variant = match.group(1)
                    current_seed = match.group(2)
                    current_epoch = 1  # reset epoch
                    print(f"\r\n-> Started new run: {current_variant} (Seed {current_seed})")
                    
            # Match epoch summary
            # Epoch 1 train_loss=0.0851 val_future_target_mae=0.289550 val_future_mae=0.289550
            if "Epoch " in line and "train_loss=" in line:
                match = re.search(r"Epoch (\d+)", line)
                if match:
                    current_epoch = int(match.group(1)) + 1
                    # Also print the summary line for the epoch
                    print(f"\r\n   Completed {line.split('INFO:src.trainers.train_cycle_model:')[-1]}")
            
            # Match batch progress
            # batch=131 loss=0.01831 lr=0.0005 param_norm=67.5085 grad_norm=0.1071
            if "batch=" in line and "loss=" in line:
                b_match = re.search(r"batch=(\d+)", line)
                l_match = re.search(r"loss=([\d\.]+)", line)
                if b_match and l_match:
                    current_batch = int(b_match.group(1))
                    current_loss = float(l_match.group(1))

if __name__ == "__main__":
    try:
        monitor_log()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
