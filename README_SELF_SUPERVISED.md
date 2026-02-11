# Self-Supervised Cycle Trading System

A reinforcement learning system that detects genuine price cycles and learns to identify confirmation patterns without optimizing for raw returns.

## Quickstart

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Generate Cache & Detect Cycles

```bash
# Fetch price data and detect cycles
python main.py
```

This will:
- Download OHLCV data for configured tickers
- Detect price cycles using deterministic rules
- Save detected cycles to `data/cycles/*.jsonl`

### 3. Build Feature Tables

```bash
python -c "from src.data.feature_engineering import assemble_features; import json; import glob; import pandas as pd; \
cycles = []; \
for f in glob.glob('data/cycles/*.jsonl'): \
    with open(f) as fp: \
        cycles.extend([json.loads(line) for line in fp]); \
features = [assemble_features(c) for c in cycles]; \
pd.DataFrame(features).to_parquet('data/feature_tables/features.parquet')"
```

### 4. Pretrain Encoder

```bash
# Run end-to-end pipeline which includes encoder pretraining
python scripts/evaluate_end_to_end.py
```

Or train encoder standalone (if you have a custom training script).

### 5. Compute Embeddings

The `evaluate_end_to_end.py` script handles this automatically. Embeddings are saved to `data/embeddings/*.parquet`.

### 6. Train RL Agent Offline

```bash
python src/trainers/train_rl.py
```

Or use the full pipeline:

```bash
python scripts/evaluate_end_to_end.py
```

### 7. Run Streaming Emulation

```bash
python simulate_live_run.py
```

This simulates real-time cycle detection and agent inference with delayed fundamental confirmations.

---

## Reward Rationale

**Core Principle**: The agent is NOT optimized for raw P&L. Instead, it learns to identify cycles that will be *confirmed by fundamentals*.

### Reward Structure

| Event | Reward |
|-------|--------|
| Flag a cycle (immediate) | `-0.01` (small cost to prevent spam) |
| Flagged cycle later confirmed | `+1.0` |
| Flagged cycle NOT confirmed | `-0.5` |
| Ignored confirmed cycle (optional) | `-0.2` |

### Why Confirm-Only Rewards?

1. **Alignment with Analysis Goal**: We want to detect cycles that represent genuine business momentum, not noise.
2. **Delayed Signal**: Fundamental confirmation arrives weeks later via earnings reports, creating a credit assignment challenge.
3. **No Direct Trading**: The environment does NOT simulate P&L because:
   - Execution costs, slippage, and regime changes make backtests unreliable
   - The agent's value is in *identifying* cycles, not trading them
   - Human discretion remains in the loop for actual trades

This design encourages the agent to learn patterns that *precede* fundamental strength rather than overfitting to price momentum alone.

---

## Example Commands

### Run Full Pipeline

```bash
python scripts/evaluate_end_to_end.py
```

### Quick Test (CI-friendly)

```bash
python scripts/evaluate_end_to_end.py --quick
```

### Train Only RL Agent

```bash
python src/trainers/train_rl.py
```

### Evaluate Trained Policy

```python
from src.trainers.train_rl import evaluate_policy_on_holdout

evaluate_policy_on_holdout('models/rl/final_model.zip')
```

### Run Online Loop Simulation

```bash
python simulate_live_run.py
```

---

## Streaming Emulation

The online loop (`src/trainers/online_loop.py`) demonstrates how the system would run in production:

1. **Cycle Detection**: Live ticks are fed to `CycleLabelStreamer`, which detects cycles in real-time.
2. **Encoding**: Detected cycles are encoded using the pretrained encoder.
3. **Inference**: The RL agent produces actions (Ignore/Flag/Proactive).
4. **Replay Buffer**: Actions are stored with cycle IDs in a replay buffer.
5. **Confirmation Backfill**: When earnings reports arrive, rewards are backfilled.
6. **Continual Learning**: The policy is periodically fine-tuned on newly confirmed cycles.

### Run Simulation

```bash
python simulate_live_run.py
```

Outputs a log to `data/live_runs/{run_id}.jsonl` showing:
- Cycle detection events
- Agent actions
- Delayed reward assignments

---

## Recommended Experiments

| Experiment | `c_flag` | `R_confirm` | `R_miss` | Embedding Dim | RL Steps |
|------------|----------|-------------|----------|---------------|----------|
| Baseline   | -0.01    | 1.0         | -0.5     | 128           | 50k      |
| High Precision | -0.05 | 1.0         | -0.8     | 128           | 50k      |
| High Recall | -0.001  | 1.0         | -0.3     | 128           | 50k      |
| Large Embed | -0.01   | 1.0         | -0.5     | 256           | 100k     |

Edit `configs/rl.yaml` to adjust hyperparameters.

---

## Directory Structure

```
.
├── configs/
│   └── rl.yaml                  # RL hyperparameters
├── data/
│   ├── cycles/                  # Detected cycles (JSONL)
│   ├── feature_tables/          # Engineered features (Parquet)
│   ├── embeddings/              # Encoded cycle embeddings (Parquet)
│   └── live_runs/               # Simulation logs
├── models/
│   ├── encoder/                 # Pretrained encoders
│   └── rl/                      # Trained RL policies
├── reports/
│   ├── figures/                 # Evaluation plots
│   ├── {run_id}.json            # Metrics (JSON)
│   └── {run_id}.md              # Human-readable report
├── scripts/
│   └── evaluate_end_to_end.py   # Full pipeline
├── src/
│   ├── data/                    # Data processing
│   ├── envs/                    # RL environment
│   ├── eval/                    # Metrics & plotting
│   ├── models/                  # Encoder architecture
│   └── trainers/                # RL training loops
├── tests/
│   └── test_end_to_end_minimal.py  # CI integration test
└── simulate_live_run.py         # Online loop demo
```

---

## Troubleshooting

**Missing Dependencies**:
```bash
pip install stable-baselines3 gymnasium torch scikit-learn matplotlib seaborn
```

**No Cycles Detected**:
- Check `config/market_universe.yaml` has valid tickers
- Ensure price data was fetched (run `main.py`)
- Verify cycle detector parameters in `src/cycle/cycle_detector.py`

**RL Training Fails**:
- Ensure embeddings exist in `data/embeddings/*.parquet`
- Check `configs/rl.yaml` for correct paths
- Reduce `total_timesteps` for faster debugging

**Plots Not Generated**:
- Install matplotlib: `pip install matplotlib seaborn`
- Check `reports/figures/` directory exists
- Verify evaluation step completed

---

## Citation

If you use this system, please cite:

```bibtex
@software{cycle_trading_rl,
  title = {Self-Supervised Cycle Trading with Delayed Fundamental Rewards},
  year = {2026},
  author = {Your Name}
}
```
