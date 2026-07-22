# Core RL Agent

This repository is currently a cycle-oriented equity modeling research pipeline. Despite the name, the active path is not the old PPO reinforcement-learning stack. The current workflow trains a PyTorch sequence model to predict cycle actions from daily technical and report-aware fundamental features.

## Active Workflow

```bash
python scripts/precompute_ground_truth.py
python src/trainers/train_cycle_model.py
# Orchestrated Phase-1 experiment (baseline -> detector-ablation -> analysis)
python scripts/orchestrate_phase1.py
python scripts/evaluate_agent.py
pytest -q
```

Default settings live in `configs/cycle_model.yaml`.

For the detailed current architecture, including the world-model framing,
memory system, input shapes, preprocessing, LSTM dimensions, targets, losses,
and outputs, see [ARCHITECTURE.md](ARCHITECTURE.md).

New tooling added for Phase-1 experiments
- `scripts/orchestrate_phase1.py` — runs the baseline training, a detector-ablation
  run (same seed/hyperparams, `configs/cycle_model_detector_ablation.yaml`), then
  runs the analysis pipeline to generate numeric reports and latent exports.
- `scripts/phase1_analysis.py` — latent PCA/UMAP projection, nearest-neighbor
  future-similarity tests, and opportunity ranking analysis. Writes summaries to
  `reports/research_phase1/` and latent exports to `reports/latent_analysis/`.

These additions implement the experiments described in the research plan without
changing model code paths or datasets: the ablation is applied by setting
`training.action_loss_weight: 0.0` in `configs/cycle_model_detector_ablation.yaml`.

## What The Project Does

1. `config/market_universe.yaml` defines the ticker universe.
2. `src/data/loader.py` fetches OHLCV, earnings dates, EPS, and revenue data.
3. `src/data/features.py` builds technical features, Minervini-style trend-template features, and report-aware fundamental features without lookahead.
4. `src/cycle/cycle_detector.py` detects heuristic price cycles.
5. `src/cycle/oracle.py` converts cycles into per-row action targets, cycle metadata, future-return targets, and price-derived event targets.
6. `src/data/sequence_dataset.py` builds rolling sequence windows and walk-forward train/validation/test splits.
7. `src/models/` defines the active sequence models and model factory.
8. `src/trainers/train_cycle_model.py` trains the model and writes a checkpoint.
9. `scripts/evaluate_agent.py` evaluates predicted action spans against oracle cycle spans.

## Current Model

The default active model is `HierarchicalLSTMCycleModel` in
`src/models/hierarchical_lstm_model.py`.

It currently uses:

- a daily LSTM over the full `252` trading-day window
- patch-level LSTM reasoning over short chunks of the daily state sequence
- attention pooling over both daily and patch-level states
- a fused latent state using daily context, patch context, latest recurrent state, and the latest raw feature vector
- a future-target regression head
- a 4-class action head
- an optional masked sequence reconstruction head used only during training

Action classes are:

- `0`: neutral
- `1`: enter
- `2`: hold_or_renew
- `3`: exit

The old PPO/Stable-Baselines RL path has been removed from the active tree. The previous Conv1D model is still available as `CycleReasoningModel` for ablation, but the active config uses the LSTM backbone. Future work can replace the LSTM with a Transformer while preserving the same model output contract:

```python
{
    "latent": latent,
    "future_pred": future_pred,
    "action_logits": action_logits,
}
```

## Repository Layout

- `config/market_universe.yaml`: ticker universe
- `configs/cycle_model.yaml`: active runtime configuration
- `scripts/precompute_ground_truth.py`: builds per-ticker feature/target parquet files
- `scripts/run_detector_on_precomputed.py`: refreshes detector labels on existing precomputed files
- `scripts/plot_detector_on_precomputed.py`: renders detector-only inspection plots
- `scripts/evaluate_agent.py`: evaluates a saved checkpoint and writes numeric metrics plus comparison plots
 - `scripts/orchestrate_phase1.py`: run baseline -> ablation -> analysis, create phase1 reports
 - `scripts/phase1_analysis.py`: latent export, PCA/UMAP, neighbor similarity, ranking tests
- `src/cycle/`: cycle detection, oracle labeling, and span decoding
- `src/data/`: data loading, feature engineering, IO, and sequence datasets
- `src/eval/`: action and cycle-level evaluation metrics
- `src/models/`: active model definitions
- `src/trainers/`: training and checkpoint logic
- `src/visualization/`: lightweight plotting helpers
- `tests/`: active pytest suite

## Generated Outputs

These are generated during runs and are ignored by git:

- `data/precomputed/*.parquet`
- `data/evaluation_plots/`
- `data/detector_plots/`
- `models/`
- `reports/`
- `logs/`
- Python cache directories

## Development Notes

The cycle detector should be treated as a weak heuristic labeler, not ground truth. The current LSTM training path includes two self-supervised/outcome-driven signals:

- random feature values and short time spans are masked during training, and the model learns to reconstruct the original standardized sequence
- event targets are derived directly from future prices: time-to-future-peak, time-to-future-drawdown, upside-before-drawdown, upside hit, and drawdown hit

The detector action loss is now separately weighted by `training.action_loss_weight`, so detector imitation can be reduced while outcome/event learning carries more of the representation. A Transformer backbone remains a later ablation once these self-supervised objectives are stable.

Run tests with:

```bash
pytest -q
```

## Phase 5 Consensus Exposure

The consensus exposure experiment keeps the transformer, market memory, C0 seed aggregation,
and A2 hold/exit rules frozen. It evaluates four predeclared gross-exposure policies using
causal percentage, volatility, and dimensionless evidence inputs. The output is development
evidence only; cross-market confirmation remains locked even when the local gates pass.

```powershell
python scripts/run_phase5_consensus_exposure.py --config configs/phase5_consensus_exposure.yaml --run-id phase5_consensus_exposure_v1
```

## Phase 5 Opportunity Allocation

This stage reconstructs causal 2022 retrieval evidence from outcome-available 2021 memory,
tests obvious versus nonlinear inference on a later 2022 slice, and then evaluates fixed-grid
per-entry sizing over the frozen 2023 C0 signals. Retrieval generation is cached and resumable.

```powershell
python scripts/run_phase5_opportunity_allocator.py --config configs/phase5_opportunity_allocator.yaml --run-id phase5_opportunity_allocator_v1 --stage all
```
