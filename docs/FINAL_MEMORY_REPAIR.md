# Closing Memory Repair Experiment

## Purpose

This is the final bounded internal architecture experiment. It repairs the two
validity defects found in `final_memory_study_v1` without reopening model
selection:

- unbounded `decision_path_quality`;
- incomplete inference coverage in the observed period.

It also enforces the repaired nanosecond causal timestamp contract and reports
per-market coverage and drawdown gates.

The experiment is diagnostic only. The 2025 through 2026 Q1 period has already
been inspected and cannot become fresh confirmation evidence.

## Frozen Contract

- Global Transformer checkpoints are reused without training.
- Global decision adapters are reused without training.
- Offline RL and reliability filtering are absent.
- Inference is regenerated once per market and seed over the complete
  2022-01-01 through 2026-03-31 interval.
- Every ticker must achieve at least 99.5% inference coverage with no more than
  one missing tail session.
- Non-finite input features are replaced only with the frozen source
  standardizer's training mean, which becomes zero after scaling.
- Standardized transfer inputs are clipped to a fixed absolute z-score of 25,
  and all repairs and clips are recorded in the export sidecar.
- The run fails if repaired cells exceed 2%, checkpoint tensors are non-finite,
  or model outputs remain non-finite.
- Retrieval requires mature outcomes with nanosecond timestamp comparison.

Path quality is:

```text
positive(MFE) / (positive(MFE) + abs(negative(MAE)) + 1e-6)
```

It is constrained to `[0, 1]` and cannot dominate fractional return evidence.

## Variants

1. `raw_c0_static`: global raw-latent C0 control.
2. `adapter_rally_bounded_static`: bounded rally-path adapter memory.
3. `adapter_rally_bounded_multiscale`: bounded rally-path growing memory over
   10, 25, and 50 neighbours.

No additional variants may be added after results are inspected.

## Execution

From the persistent Jarvis repository:

```bash
cd /home/core-rl-phase6
git pull --ff-only origin main
source .venv/bin/activate
RUN_ID=final_memory_repair_v1
```

Build the durable DAG:

```bash
.venv/bin/python scripts/run_final_memory_study.py \
  --config configs/final_memory_repair.yaml \
  --run-id "$RUN_ID" \
  --stage build \
  --python .venv/bin/python
```

Inspect the plan:

```bash
.venv/bin/python scripts/run_durable_experiment.py \
  --manifest "reports/final_memory_repair/$RUN_ID/experiment_manifest.yaml" \
  --state-dir "reports/final_memory_repair/$RUN_ID/orchestration_state" \
  --project-root /home/core-rl-phase6 \
  --plan-summary
```

The plan must report 265 jobs, 18 GPU inference-refresh jobs, three variants,
zero training jobs, zero RL jobs, and a 5.4 GPU-hour declared upper bound.

Run or resume:

```bash
.venv/bin/python scripts/run_durable_experiment.py \
  --manifest "reports/final_memory_repair/$RUN_ID/experiment_manifest.yaml" \
  --state-dir "reports/final_memory_repair/$RUN_ID/orchestration_state" \
  --project-root /home/core-rl-phase6
```

## Decision Rule

The bounded multiscale mechanism survives only if it improves over raw C0 while
also passing all of:

- pooled Sharpe at least 2.0;
- pooled drawdown no worse than 20%;
- every market drawdown no worse than 25%;
- at least five profitable markets;
- at least four equal-weight Sharpe wins;
- median market Sharpe at least 0.50;
- minimum market inference coverage at least 99.5%;
- profit concentration no greater than 35%;
- non-negative seed-ensemble lift.

Because all periods have already been inspected, passing these gates would
justify a fresh external confirmation, not a profitable-strategy claim.
