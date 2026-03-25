# Cycle Reasoning System

This branch now uses a sequence model for cycle reasoning instead of the older binary policy stack.

## Core Idea

- Detect oracle cycles retroactively from price windows.
- Align report-aware fundamentals without leakage.
- Train a multi-scale causal 1D model over trend features.
- Predict `neutral / enter / hold_or_renew / exit`.
- Score results at the cycle level, with profitable start-to-end moves treated as the base goodness signal.

## Default Flow

```bash
python scripts/precompute_ground_truth.py
python scripts/audit_precomputed_data.py
python src/trainers/train_cycle_model.py
python scripts/evaluate_agent.py
```

## Model

The active model is a multi-scale temporal encoder:

- `21d / 63d / 252d` causal convolution branches
- fused latent state for trend reasoning
- auxiliary future-summary regression head
- 4-action decision head

This keeps the JEPA-like predictive-latent flavor without using periodicity-to-image indirection.

## Main Files

- `configs/cycle_model.yaml`: training, split, and evaluation config
- `scripts/precompute_ground_truth.py`: builds daily feature tables plus oracle action targets
- `src/trainers/train_cycle_model.py`: trains the sequence model and writes reports
- `scripts/evaluate_agent.py`: evaluates a saved checkpoint and writes comparison plots

## Outputs

- `data/precomputed/*.parquet`: daily feature tables with oracle targets
- `models/cycle_reasoner/final_model.pt`: trained checkpoint
- `reports/experiments/*.json|*.md`: clean experiment summaries
- `data/evaluation_plots/*_comparison.png`: oracle vs predicted cycle plots
