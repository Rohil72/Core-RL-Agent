# Final Memory Confirmation

## Evidence Boundary

The confirmation evaluates the frozen `global_global__coverage_025` candidate
from `configs/final_memory_policy.yaml`. The candidate failed its development
promotion gate. Confirmation is therefore an external robustness test, not a
retroactive promotion or permission to tune against 2025-2026.

The protocol is fixed in `configs/final_memory_confirmation.yaml`:

- Global Transformer, global internal memory, and global external memory.
- Seeds 7, 17, and 37 with two-vote consensus.
- Reliability model fitted only on the existing 2022-2023 development evidence.
- Locked 25% reliability-coverage rule.
- Confirmation queries from 2025-01-01 through 2026-03-31.
- Outcomes matured through 2026-06-30.
- No encoder, adapter, memory, threshold, or policy training.
- No RL.

## Baselines

Every market is evaluated with the same chronology and declared execution
costs. The report includes:

- Equal-weight buy-and-hold with one entry and exit cost.
- Raw consensus memory without reliability filtering.
- Cross-sectional 21-session momentum.
- Direct adapter median `pred_utility_q50`.
- Twenty deterministic random-ranking trials.

Momentum, direct-adapter, and random rankings use the same top-k, holding,
stop-loss, risk-ledger, and transaction-cost engine as the memory policy. They
do not inherit memory confidence, downside, or neighbour gates.

## One-Shot Lock

Building the run hashes the confirmation configuration, final policy,
requirements, evaluator and memory source, global checkpoints and adapters,
development evidence, global historical memories, and all six market datasets.
An existing unexecuted lock can be reused only when every immutable input is
identical. An executed lock cannot be replaced.

Use a unique run ID:

```bash
cd /home/core-rl-phase6
git pull --ff-only origin main
source .venv/bin/activate

RUN_ID=final_memory_confirmation_v1

.venv/bin/python scripts/run_final_memory_confirmation.py \
  --config configs/final_memory_confirmation.yaml \
  --run-id "$RUN_ID" \
  --stage build \
  --python .venv/bin/python
```

Inspect the manifest before execution:

```bash
.venv/bin/python scripts/run_durable_experiment.py \
  --manifest "reports/final_memory_confirmation/$RUN_ID/experiment_manifest.yaml" \
  --state-dir "reports/final_memory_confirmation/$RUN_ID/orchestration_state" \
  --project-root /home/core-rl-phase6 \
  --plan-summary
```

The manifest contains 44 jobs: 18 frozen encoder exports, 18 causal retrieval
jobs, six market evaluations, one aggregate, and one lock close. Only the 18
exports use CUDA. The declared 5.4 GPU-hour value is a conservative job upper
bound, not a runtime or spending prediction.

Run or resume with the same command:

```bash
.venv/bin/python scripts/run_durable_experiment.py \
  --manifest "reports/final_memory_confirmation/$RUN_ID/experiment_manifest.yaml" \
  --state-dir "reports/final_memory_confirmation/$RUN_ID/orchestration_state" \
  --project-root /home/core-rl-phase6
```

## Outputs

The primary outputs are:

```text
reports/final_memory_confirmation/<run-id>/confirmation_summary.json
reports/final_memory_confirmation/<run-id>/strategy_summary.csv
reports/final_memory_confirmation/<run-id>/market_baseline_results.csv
reports/final_memory_confirmation/<run-id>/calibration_results.csv
reports/final_memory_confirmation/<run-id>/confirmation_report.md
reports/final_memory_confirmation/<run-id>/confirmation_lock.json
```

Market directories retain trades, equity curves, decisions, consensus signals,
reliability predictions, model audits, random-trial distributions, and
per-baseline metrics.

## Interpretation

The aggregate report explicitly includes pooled Sharpe and return, maximum
drawdown, positive markets, equal-weight wins, Brier-skill breadth, deflated
Sharpe probability, profit concentration, and the dominant profit market.

Passing the external gates is evidence of robustness. It does not erase the
failed development gate, survivorship bias, fixed-universe selection, or the
fact that this candidate was chosen after a multi-variant research programme.
The report therefore never automatically permits a profitable-strategy claim.
