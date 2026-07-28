# Local Rank Ensemble

## Purpose

This protocol reconstructs the mechanism associated with the strongest earlier
Phase 5 development result. It is not a continuation of the failed
reliability-gated confirmation and it cannot overwrite that result.

The reconstruction changes only the lightweight reasoning layer:

```text
existing regional encoder checkpoint
  -> market-local 128 -> 64 -> 32 decision adapter
  -> expanding market-local causal memory
  -> three independent seed percentile ranks
  -> cross-seed median rank
  -> deterministic top-k portfolio
```

There is no transformer training, offline RL, binary vote requirement, or
reliability filter.

## Fixed Contract

`configs/local_rank_ensemble.yaml` declares:

- Six markets and seeds 7, 17, and 37.
- A 32-dimensional normalized decision space.
- Quantile, within-date ranking, analogue geometry, and VICReg losses.
- A 63-session primary opportunity horizon.
- Twenty-five causal cross-ticker neighbours.
- Gaussian robust evidence aggregation.
- Zero required seed votes and median percentile-rank aggregation.
- Market-specific transaction-cost assumptions.
- Equal-weight, 21-session momentum, direct-adapter, repeated-random, and
  individual-seed baselines.

The adapter is the only trained component. It reuses each
`regional_<market>_seed_<seed>` encoder checkpoint and latent export from
`phase6_a30_final_v1`.

## Causal Growing Memory

For each period, memory contains adapter training decisions plus decisions
exported through that period. A row can be retrieved only when:

```text
neighbor.outcome_available_timestamp <= query.timestamp
```

Rows without a known maturity timestamp fail closed. The query period may be
present in the physical memory table because the retrieval layer enforces this
per-query availability rule. This permits memory to grow without revealing a
future outcome.

## Evidence Boundary

The periods have different meanings:

- `development`: 2022-2023 development evidence.
- `selection`: 2024 development-selection evidence.
- `observed_confirmation`: 2025 through 2026 Q1, already inspected and therefore
  diagnostic only.

Every aggregate result writes `promotion_allowed: false`. Passing numerical
gates means the mechanism deserves an untouched future test; it is not fresh
confirmation.

## Build And Run

On the persistent Jarvis checkout, after pulling the commit and activating the
existing Python 3.11 environment:

```bash
cd /home/core-rl-phase6
RUN_ID=local_rank_ensemble_v1

.venv/bin/python scripts/run_local_rank_ensemble.py \
  --config configs/local_rank_ensemble.yaml \
  --run-id "$RUN_ID" \
  --stage build \
  --python .venv/bin/python

.venv/bin/python scripts/run_durable_experiment.py \
  --manifest "reports/local_rank_ensemble/$RUN_ID/experiment_manifest.yaml" \
  --state-dir "reports/local_rank_ensemble/$RUN_ID/orchestration_state" \
  --project-root /home/core-rl-phase6 \
  --stage full \
  --plan-summary

.venv/bin/python scripts/run_durable_experiment.py \
  --manifest "reports/local_rank_ensemble/$RUN_ID/experiment_manifest.yaml" \
  --state-dir "reports/local_rank_ensemble/$RUN_ID/orchestration_state" \
  --project-root /home/core-rl-phase6 \
  --stage full
```

The durable runner skips completed artifacts and resumes the same manifest after
preemption. Do not rebuild with changed source or configuration under the same
run ID.

## Outputs

The principal artifacts are:

```text
reports/local_rank_ensemble/<run-id>/
  adapters/
  decisions/
  growing_memory/
  memory/
  evaluation/<period>/<market>/
  aggregate/<period>/market_results.csv
  aggregate/<period>/strategy_summary.csv
  aggregate/<period>/summary.json
  experiment_manifest.yaml
  orchestration_state/
```

Each market evaluation includes consensus signals, all trades and equity
curves, repeated-random trials, individual-seed results, evidence correlations,
and ensemble Sharpe lift versus the mean seed.
