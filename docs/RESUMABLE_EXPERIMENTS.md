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

## Locked final memory policy

The active final configuration is `configs/final_memory_policy.yaml`. It
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

The compiled canonical DAG contains one CPU evaluation job. Reissuing the final
command is safe: completed outputs are validated and skipped.

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
