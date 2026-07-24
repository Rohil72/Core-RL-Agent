# Core RL Agent

Core RL Agent is a research codebase for causal, retrieval-grounded equity
decisions. It is not a live-trading system and no policy is currently promoted.
The active path is a patch Transformer market-state encoder, a historical
market-memory layer, and offline policies that choose portfolio exposure from
retrieved evidence.

The cycle detector remains only for earlier experiments. The final encoder uses
`patch_transformer`, disables detector targets, and sets detector action loss to
zero.

## Research Status

Development results support further work on the frozen latent representation and
historical memory, but they do not support a performance claim. Several Phase 5
policy/allocation selectors were explicitly rejected by their robustness gates.
The current task is the frozen six-market regional/global/RL testbed, not more
tuning on those single-market development results.

See [Research Status](docs/RESEARCH_STATUS.md) for the evidence boundary,
rejected hypotheses, promotion standard, and known limitations.

```text
daily technical and point-in-time fundamental features
  -> 252-session patch Transformer
  -> 128-dimensional latent state
  -> causal market memory and analogue evidence
  -> offline exposure policy
  -> market-level and pooled robustness gates
```

## Clean-Start Setup

Python 3.11 and a CUDA-capable NVIDIA GPU are required for the final testbed.
The encoder and offline-RL stack use separate environments.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

py -3.11 -m venv .venv-phase5-rl
.\.venv-phase5-rl\Scripts\python.exe -m pip install --upgrade pip
.\.venv-phase5-rl\Scripts\python.exe -m pip install -r requirements-phase5-rl.txt
```

Compile the final DAG on the machine that will execute it:

```powershell
.\.venv\Scripts\python.exe scripts\build_final_testbed.py `
  --config configs\final_research_testbed.yaml `
  --base-encoder configs\final_encoder_training.yaml `
  --run-id phase6_international_v1 `
  --core-python .venv\Scripts\python.exe `
  --rl-python .venv-phase5-rl\Scripts\python.exe
```

The stage-by-stage execution and confirmation-lock procedure is in
[Resumable Experiments](docs/RESUMABLE_EXPERIMENTS.md).

## Final Testbed

`configs/final_research_testbed.yaml` fixes the current protocol:

- Encoder: train 2013-2020, validate 2021, test 2022.
- Policy development: 2022-2023; selection: 2024.
- Locked confirmation: 2025-01-01 through 2026-03-31, with outcomes matured
  through 2026-06-30.
- Markets: US, India, China, Brazil, France, and UK.
- Representations: regional encoder and pooled global encoder.
- Policies: contextual bandit, CQL, IQL, and TD3+BC. A four-market pilot picks
  exactly two algorithms before a six-market, three-seed comparison.
- Promotion: Sharpe >= 2.0, drawdown <= 20%, at least five positive markets,
  bounded profit concentration, non-degenerate exposure, and locked confirmation.

The testbed requires CUDA, one GPU with at least 6 GB VRAM, and limits the
process to 80% of selected GPU VRAM. `minimum_free_gb: 60` is a conservative
start-time guard, not a predicted disk requirement.

## Repository Map

- `src/models/patch_transformer_model.py`: active encoder.
- `src/losses/outcome_geometry.py`: regression and optional geometry losses.
- `src/memory/`: causal retrieval, evidence aggregation, and confidence.
- `src/decision/`: counterfactual decision outcomes.
- `src/policy/`: offline-policy dataset and contextual bandit.
- `src/backtest/`: memory-policy backtesting.
- `src/trainers/train_cycle_model.py`: encoder training and resumable state.
- `src/orchestration/`: immutable durable experiment runner.
- `scripts/build_final_testbed.py`: final international DAG compiler.
- `scripts/build_confirmation_testbed.py`: post-promotion confirmation compiler.

Generated data, checkpoints, models, reports, virtual environments, and
Graphify outputs are ignored by Git. A clean clone contains runnable code,
configuration, tests, and the small sample CSV fixture only.

## Documentation

- [Architecture](ARCHITECTURE.md)
- [Research Status](docs/RESEARCH_STATUS.md)
- [Resumable Experiments](docs/RESUMABLE_EXPERIMENTS.md)
