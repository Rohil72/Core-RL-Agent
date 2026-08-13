# Resumable Experiments

Long encoder and policy experiments must run through
`scripts/run_durable_experiment.py`. The runner treats the manifest, source,
input data, Python/PyTorch/CUDA stack, driver, and accelerator class as one
immutable experiment contract.
Each Python executable named by a job is probed separately, so the core encoder
environment and isolated RL environment are both pinned.

## Manifest schema

Commands are argument lists, not shell strings. Paths are relative to the
project root unless they are absolute. Put the state directory and all declared
outputs on persistent storage.

```yaml
run:
  id: international_pilot_v1
  strict_environment: true
  stale_lock_seconds: 900
  inputs:
    - data/international/**/*.parquet
  source_patterns:
    - src/**/*.py
    - scripts/**/*.py
    - configs/**/*.yaml
    - requirements*.txt

jobs:
  - id: encoder_US_seed_7
    command:
      - python
      - -m
      - src.trainers.train_cycle_model
      - --config-path
      - reports/phase5/international/pilot/training_configs/US/seed_7.yaml
      - --resume
    inputs:
      - reports/phase5/international/pilot/training_configs/US/seed_7.yaml
      - data/international/US/*.parquet
    expected_outputs:
      - reports/phase5/international/pilot/models/US/seed_7/final_model.pt
      - reports/phase5/international/pilot/models/US/seed_7/training_complete.json

  - id: build_US_memory_seed_7
    depends_on: [encoder_US_seed_7]
    command: [python, scripts/example_memory_command.py]
    expected_outputs:
      - reports/phase5/international/pilot/memory/US/seed_7/metrics.json
```

The final research manifest is generated from
`configs/final_research_testbed.yaml`; the fragment above only illustrates the
durable-runner schema.

## Usage

Inspect the dependency graph without executing commands:

```powershell
python scripts\run_durable_experiment.py `
  --manifest <manifest.yaml> `
  --state-dir <persistent-output>\.experiment_state `
  --plan
```

Start or resume the run with the same command minus `--plan`:

```powershell
python scripts\run_durable_experiment.py `
  --manifest <manifest.yaml> `
  --state-dir <persistent-output>\.experiment_state
```

Re-running this command skips jobs only when their completion state and every
declared output digest are valid. A job left in `running` or `interrupted` state
is run again. Encoder training then resumes from `latest_training_state.pt`.
Changing a completed artifact causes a hard integrity failure rather than a
silent rerun.

## Encoder continuation contract

With `--resume`, the trainer writes an atomic continuation checkpoint every 250
optimizer steps and after every validation epoch. The checkpoint contains:

- current model and optimizer state;
- AMP gradient-scaler state;
- best validation model and score;
- Python, NumPy, CPU Torch, CUDA, and data-loader RNG states;
- exact epoch and next batch position;
- partial epoch loss accounting;
- immutable configuration and runtime fingerprints.

`SIGINT`, `SIGTERM`, and Windows `SIGBREAK` request a checkpoint after the
current optimizer step. The process then exits non-zero so the outer runner
records an interruption rather than a false completion.

Do not set `allow_hardware_mismatch_resume` for comparable experiments. If a
machine must change, start a new run ID and treat it as a separate testbed.

## Active final memory study

The active CPU-only DAG is generated from
`configs/final_memory_study.yaml`. It reuses the completed
`local_rank_ensemble_v1` adapter and decision artifacts and contains no
Transformer, adapter-training, or RL jobs:

```powershell
$run = "final_memory_study_v1"

.\.venv\Scripts\python.exe scripts\run_final_memory_study.py `
  --config configs\final_memory_study.yaml `
  --run-id $run `
  --stage build `
  --python .venv\Scripts\python.exe

$manifest = "reports\final_memory_study\$run\experiment_manifest.yaml"
$state = "reports\final_memory_study\$run\orchestration_state"

.\.venv\Scripts\python.exe scripts\run_durable_experiment.py `
  --manifest $manifest `
  --state-dir $state `
  --stage full `
  --plan-summary

.\.venv\Scripts\python.exe scripts\run_durable_experiment.py `
  --manifest $manifest `
  --state-dir $state `
  --stage full
```

The declared graph contains 322 jobs and zero GPU-hour allowance. Reissuing the
last command validates and skips completed artifacts. See
[Final Memory Study](FINAL_MEMORY_STUDY.md) for the variant and evidence
contract.

## Historical local-rank reconstruction

The rejected exploratory DAG is generated from
`configs/local_rank_ensemble.yaml`. It reuses the ignored regional Phase 6
encoder checkpoints and latent exports, trains only lightweight decision
adapters, and contains no transformer or RL jobs:

```powershell
$run = "local_rank_ensemble_v1"

.\.venv\Scripts\python.exe scripts\run_local_rank_ensemble.py `
  --config configs\local_rank_ensemble.yaml `
  --run-id $run `
  --stage build `
  --python .venv\Scripts\python.exe

$manifest = "reports\local_rank_ensemble\$run\experiment_manifest.yaml"
$state = "reports\local_rank_ensemble\$run\orchestration_state"

.\.venv\Scripts\python.exe scripts\run_durable_experiment.py `
  --manifest $manifest `
  --state-dir $state `
  --stage full `
  --plan-summary

.\.venv\Scripts\python.exe scripts\run_durable_experiment.py `
  --manifest $manifest `
  --state-dir $state `
  --stage full
```

Reissuing the final command validates and skips completed jobs. The declared
graph contains 202 jobs and at most 34.2 estimated GPU-hours. See
[Local Rank Ensemble](LOCAL_RANK_ENSEMBLE.md) for its causal and evidentiary
contract.

## Historical locked memory policy

The rejected locked configuration is `configs/final_memory_policy.yaml`. It
replays one preselected global/global memory candidate and contains no encoder
training or RL jobs. Restore the ignored `phase6_a30_final_v1` artifacts, then
compile and execute:

```powershell
$run = "final_memory_policy_v1"

.\.venv\Scripts\python.exe scripts\run_phase6_frozen_memory_sweep.py `
  --config configs\final_memory_policy.yaml `
  --run-id $run `
  --stage build `
  --python .venv\Scripts\python.exe

$manifest = "reports\final_memory_policy\$run\experiment_manifest.yaml"
$state = "reports\final_memory_policy\$run\orchestration_state"

.\.venv\Scripts\python.exe scripts\run_durable_experiment.py `
  --manifest $manifest `
  --state-dir $state `
  --plan-summary

.\.venv\Scripts\python.exe scripts\run_durable_experiment.py `
  --manifest $manifest `
  --state-dir $state
```

The compiled historical DAG contains one CPU evaluation job. Reissuing the final
command is safe: completed outputs are validated and skipped.

## Historical sealed external evaluation

The completed failed-confirmation procedure used
`configs/final_memory_confirmation.yaml` and
`scripts/run_final_memory_confirmation.py` for the one-shot 2025-2026 Q1
evaluation. It runs frozen inference, causal retrieval, five predeclared
baseline families, calibration diagnostics, concentration reporting, and
closes an immutable confirmation lock. Complete commands and the evidence
boundary are in [Final Memory Confirmation](FINAL_MEMORY_CONFIRMATION.md).

## Historical international RL testbed

The frozen research schema is `configs/final_research_testbed.yaml`. It defines
the six local markets, 2013-2026 chronology, regional/global encoders, pilot RL
gate, three final seeds, CUDA allowance, persistent storage requirements, and
stage GPU-hour budgets.

Create both environments before compiling the manifest. On Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

py -3.11 -m venv .venv-phase5-rl
.\.venv-phase5-rl\Scripts\python.exe -m pip install --upgrade pip
.\.venv-phase5-rl\Scripts\python.exe -m pip install -r requirements-phase5-rl.txt

.\.venv\Scripts\python.exe scripts\build_final_testbed.py `
  --config configs\final_research_testbed.yaml `
  --run-id phase6_international_v1 `
  --core-python .venv\Scripts\python.exe `
  --rl-python .venv-phase5-rl\Scripts\python.exe
```

For a metered cloud instance, optionally hard-cap spend while compiling. For
example, using the displayed A100 rate:

```powershell
  --hourly-cost-inr 85.37 --pilot-max-cost-inr 1500 --full-max-cost-inr 6000
```

Hourly pricing is deliberately supplied at build time rather than frozen in
source control. The runner checks projected and consumed stage GPU-hours before
starting every GPU job.

On a Linux cloud instance, use `.venv/bin/python` and
`.venv-phase5-rl/bin/python` for the final two arguments. Compile the manifest
on the machine that will execute it so interpreter paths are correct.

Run one stage at a time:

```powershell
$manifest = "reports\final_testbed\phase6_international_v1\experiment_manifest.yaml"
$state = "reports\final_testbed\phase6_international_v1\orchestration_state"

.\.venv\Scripts\python.exe scripts\run_durable_experiment.py --manifest $manifest --state-dir $state --stage data
.\.venv\Scripts\python.exe scripts\run_durable_experiment.py --manifest $manifest --state-dir $state --stage pilot
.\.venv\Scripts\python.exe scripts\run_durable_experiment.py --manifest $manifest --state-dir $state --stage full
```

The pilot stage must select exactly two algorithms across US, India, China, and
Brazil. The full stage then compares only those algorithms with regional and
global representations across all six markets and seeds 7, 17, and 37.

The confirmation stage is intentionally not generated. After
`selection/final_policy_selection.json` passes, create the one-shot confirmation
lock and compile its separate manifest:

```powershell
.\.venv\Scripts\python.exe scripts\build_confirmation_testbed.py `
  --config configs\final_research_testbed.yaml `
  --run-id phase6_international_v1 `
  --development-state $state `
  --core-python .venv\Scripts\python.exe `
  --rl-python .venv-phase5-rl\Scripts\python.exe

.\.venv\Scripts\python.exe scripts\run_durable_experiment.py `
  --manifest reports\final_testbed\phase6_international_v1\confirmation\confirmation_manifest.yaml `
  --state-dir reports\final_testbed\phase6_international_v1\confirmation\orchestration_state `
  --stage confirmation
```

The confirmation compiler refuses to run without a promoted development
candidate. It hashes every selected encoder, adapter, policy, dataset, and
international market-data file, then pins both the development source tree and
CUDA/PyTorch runtime. This prevents accidental use of 2025 through 2026 Q1
while development is still changing or on a different testbed.

The repository, `data/international`, `reports/final_testbed`, and the state
directory must live on storage that survives instance termination. Graceful
preemption saves after the current optimizer step; abrupt loss resumes from the
latest periodic checkpoint.

## Audited spot-instance runtime migration

The strict runtime contract intentionally stops a resumed run when the host
fingerprint changes. A spot instance may return with a different Linux kernel
release even though Python, PyTorch, CUDA, cuDNN, the NVIDIA driver, GPU model,
compute capability, and VRAM are unchanged. Accept only that declared host
metadata change:

```bash
.venv/bin/python scripts/run_durable_experiment.py \
  --manifest "reports/temporal_transport_encoder/$RUN_ID/experiment_manifest.yaml" \
  --state-dir "reports/temporal_transport_encoder/$RUN_ID/orchestration_state" \
  --project-root /home/core-rl-phase6 \
  --migrate-runtime \
  --allow-runtime-field release \
  --migration-reason "Spot instance resumed with a different Linux kernel release; Python, PyTorch, CUDA, cuDNN, driver and A30 hardware are identical." \
  --migration-operator rohil
```

The migration writes a chained record under `orchestration_state/migrations`,
preserves completed jobs, and updates the runtime fingerprint. It refuses ML
runtime, source, manifest, input, or job-environment changes. Resume with the
ordinary runner command after the migration succeeds.

When pulling an operational runner fix and returning on a changed spot-instance
kernel at the same time, migrate both declared changes atomically:

```bash
.venv/bin/python scripts/run_durable_experiment.py \
  --manifest "reports/temporal_transport_encoder/$RUN_ID/experiment_manifest.yaml" \
  --state-dir "reports/temporal_transport_encoder/$RUN_ID/orchestration_state" \
  --project-root /home/core-rl-phase6 \
  --migrate-source \
  --allow-source-path scripts/run_durable_experiment.py \
  --allow-source-path scripts/export_manuscript_evidence.py \
  --allow-source-path src/orchestration/durable_runner.py \
  --allow-runtime-field release \
  --migration-reason "Install the audited runtime-resume and evidence-export utilities while accepting only the spot-instance Linux kernel release change." \
  --migration-operator rohil
```

## Manuscript evidence archive

Create a compact reviewer archive after comparison completes:

```bash
.venv/bin/python scripts/export_manuscript_evidence.py \
  --run-root "reports/temporal_transport_encoder/$RUN_ID" \
  --output "/home/temporal_transport_v2_reviewer_evidence.tar.gz" \
  --profile reviewer
```

Repeat `--run-root` to consolidate related historical testbeds. The reviewer
profile includes contracts, migrations, resolved configurations, comparisons,
metrics, trades, signals, neighbours, training summaries, audits, provenance,
and SHA-256 checksums where available. It also writes `missing_evidence.json`;
absence is never treated as proof.

For private reproducibility storage, the full profile additionally includes
latent tables, checkpoints, job states, and logs and can therefore be large:

```bash
.venv/bin/python scripts/export_manuscript_evidence.py \
  --run-root "reports/temporal_transport_encoder/$RUN_ID" \
  --output "/home/temporal_transport_v2_full_evidence.tar.gz" \
  --profile full
```

Do not upload the full profile to a public repository without checking data
provider terms and archive size. The reviewer archive supports manuscript
claims; it does not cure survivorship bias, create point-in-time constituents,
or turn an observed interval into untouched confirmation data.
