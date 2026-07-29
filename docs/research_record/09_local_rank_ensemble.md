# Local Rank Ensemble Reconstruction

`[REJECTED] [NEEDS-REPAIR]`

## Purpose

This run reconstructed the strongest earlier Phase 5 mechanism without RL:
regional frozen encoders, market-local 128-to-64-to-32 adapters, growing causal
memory, and three-seed continuous rank aggregation.

## Recorded Run

The run completed 202/202 jobs. Its aggregate evidence was:

| Period | Return | Sharpe | Drawdown |
|---|---:|---:|---:|
| Development | -2.90% | -0.046 | -20.18% |
| Selection | -0.98% | -0.062 | not retained in the compact audit summary |
| Observed | +3.56% | 0.318 | not retained in the compact audit summary |

Only Brazil was positive in the observed period. Memory beat the direct head in
13 of 18 market-period comparisons but beat equal-weight in only 4 of 18.
Mean memory-alpha Spearman was approximately `0.007`, `0.021`, and `0.064`
across the three periods; interval coverage was approximately 16%.

Ensemble lift was negative across all periods and the ensemble beat the best
individual seed in only 3 of 18 comparisons.

## Critical Audit Finding

The memory frames contained both `latent_*` and `decision_*` columns. The
generic loader selected `latent_*`, so retrieval used the raw 128-dimensional
encoder geometry while the direct-head comparison used the trained 32D adapter
space. This meant the run did not cleanly test adapter-space retrieval.

The final memory study was created to isolate and repair that ambiguity.

## Source Map

- `configs/local_rank_ensemble.yaml`
- `scripts/run_local_rank_ensemble.py`
- `src/eval/local_rank_ensemble.py`
- `reports/local_rank_ensemble/local_rank_ensemble_v1/` when restored
