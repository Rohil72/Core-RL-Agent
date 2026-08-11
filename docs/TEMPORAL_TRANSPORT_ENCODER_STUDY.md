# Temporal Transport Encoder Study

## Question

This is the final representation-level diagnostic:

> Can explicitly preserving outcome-neighbourhood semantics across years and
> markets prevent the retrieval signal from disappearing under temporal
> transfer?

The study does not use RL and does not add confidence or calibration gates. It
uses the deterministic C2 policy with a minimum 21-session hold.

## Controlled Comparison

Both variants use the same patch Transformer, input window, targets, training
period, seeds, optimizer, memory, policy, and transaction costs.

- `baseline`: masked Huber future-outcome regression.
- `temporal_transport`: the same regression plus cross-market/cross-period
  analogue KL, temporal hard-negative triplets, and a small anti-collapse term.

The training data spans ten years, validation and test each consume the next
year, and checkpoint selection uses validation MAE only. The configured
`observed` interval (`2025-01-01` through `2026-03-31`) is evaluated only after
the best checkpoint is frozen. It is already contaminated by prior research
inspection, so the output is diagnostic and cannot be promoted as fresh
confirmation.

## Transport Objective

For each valid anchor, analogue candidates must:

1. come from another ticker;
2. come from another market;
3. be separated by at least two calendar years.

The analogue KL matches latent-neighbour probabilities to benchmark-relative
future-outcome similarity. Candidate probability mass is balanced by
market-year domain. The triplet term pulls the closest cross-domain outcome
analogue toward the anchor and pushes away the nearest latent hard negative
whose future outcome lies in the adverse outcome-distance tail.

The target geometry contains 63-session market-relative alpha, upside, downside,
and upside-before-drawdown path quality. Market-relative technical features are
computed point-in-time within each country and date. Future alpha is also
benchmarked within country, preventing one market's base rally rate from being
presented as stock-selection skill.

## Acceptance Boundary

The candidate passes the 2024 development-test gate only when it:

- beats the baseline Sharpe in at least two of three paired seeds;
- has positive mean Sharpe lift;
- does not regress in the worst seed; and
- keeps worst drawdown within 25%.

Even a pass selects only a mechanism for a later frozen/live test. A fail ends
the profitable-transfer branch: further inference thresholds, recency kernels,
calibration layers, and RL policies are out of scope.

## Durable Execution

Build the six training jobs, twelve period-specific evaluations, and paired comparison:

```bash
RUN_ID=temporal_transport_v1

.venv/bin/python scripts/run_temporal_transport_study.py \
  --config configs/temporal_transport_study.yaml \
  --run-id "$RUN_ID" \
  --stage build \
  --python .venv/bin/python
```

Inspect the immutable DAG:

```bash
.venv/bin/python scripts/run_durable_experiment.py \
  --manifest "reports/temporal_transport_encoder/$RUN_ID/experiment_manifest.yaml" \
  --state-dir "reports/temporal_transport_encoder/$RUN_ID/orchestration_state" \
  --project-root /home/core-rl-phase6 \
  --plan-summary
```

Run or resume by removing `--plan-summary`. Each trainer uses optimizer-level
checkpoints, and completed jobs are not repeated.

## Outputs

- `comparison/run_results.csv`: one 2024-test and observed-period result per
  variant and seed.
- `comparison/variant_summary.csv`: aggregate return, Sharpe, drawdown, trades,
  and score-realized-return correlation.
- `comparison/paired_results.csv`: exact seed-matched candidate deltas.
- `comparison/verdict.json`: predeclared gate result with promotion disabled.
- Per-run checkpoints, latent exports, signals, neighbors, trades, and equity
  curves remain under the same immutable run directory.
