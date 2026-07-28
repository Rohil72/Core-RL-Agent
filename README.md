# Core RL Agent

Core RL Agent is a research codebase for causal, retrieval-grounded equity
decisions. It is not a live-trading system and no policy is currently promoted.
The active path is a patch Transformer market-state encoder, market-local
historical memory, and a deterministic continuous-rank policy that converts
three independent retrieval views into trades. Offline RL remains available
only as a rejected Phase 6 comparison path.

The cycle detector remains only for earlier experiments. The final encoder uses
`patch_transformer`, disables detector targets, and sets detector action loss to
zero.

## Research Status

The sealed global/global reliability candidate failed confirmation and remains
immutable. A separate exploratory reconstruction now tests the strongest
mechanism seen in the earlier Phase 5 development result: regional encoders,
market-local decision adapters and growing memory, and an ungated three-seed
median-rank policy. It does not reopen or overwrite the failed confirmation,
and no policy is promoted.

See [Research Status](docs/RESEARCH_STATUS.md) for the evidence boundary,
rejected hypotheses, promotion standard, and known limitations.

```text
daily technical and point-in-time fundamental features
  -> 252-session patch Transformer
  -> 128-dimensional latent state
  -> causal market memory and analogue evidence
  -> market-local growing causal memory
  -> continuous three-seed median-rank policy
  -> market-level and pooled robustness gates
```

## Clean-Start Setup

Python 3.11 is required. CUDA is required only when training encoders; replaying
the locked final memory-policy evaluation is CPU-capable.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Compile the local-rank reconstruction after restoring the ignored Phase 6
regional checkpoints and latent exports:

```powershell
.\.venv\Scripts\python.exe scripts\run_local_rank_ensemble.py `
  --config configs\local_rank_ensemble.yaml `
  --run-id local_rank_ensemble_v1 `
  --stage build `
  --python .venv\Scripts\python.exe
```

The stage-by-stage execution and confirmation-lock procedure is in
[Resumable Experiments](docs/RESUMABLE_EXPERIMENTS.md).

## Active Testbed

`configs/local_rank_ensemble.yaml` fixes the exploratory reconstruction:

- Existing regional patch Transformer checkpoints; no encoder retraining.
- Market-local 128-to-64-to-32 decision adapters.
- Growing causal memory that admits an outcome only after it matures.
- Three-seed continuous median rank with zero required binary votes.
- Deterministic top-k trading policy and declared standard baselines.
- Markets: US, India, China, Brazil, France, and UK.
- No RL and no reliability filter.
- Development, selection, and already-observed diagnostic periods are reported
  separately. None can create a fresh confirmation claim.

`configs/final_research_testbed.yaml` and `scripts/build_final_testbed.py`
preserve the completed regional/global/offline-RL experiment for reproducibility,
but they are no longer the active execution path.

The reconstruction contract and commands are documented in
[Local Rank Ensemble](docs/LOCAL_RANK_ENSEMBLE.md). The failed sealed procedure
remains documented in
[Final Memory Confirmation](docs/FINAL_MEMORY_CONFIRMATION.md).

## Repository Map

- `src/models/patch_transformer_model.py`: active encoder.
- `src/losses/outcome_geometry.py`: regression and optional geometry losses.
- `src/memory/`: causal retrieval, evidence aggregation, and confidence.
- `src/decision/`: counterfactual decision outcomes.
- `src/policy/`: deterministic policy plus historical offline-RL comparisons.
- `src/backtest/`: memory-policy backtesting.
- `src/trainers/train_cycle_model.py`: encoder training and resumable state.
- `src/orchestration/`: immutable durable experiment runner.
- `scripts/run_local_rank_ensemble.py`: active exploratory DAG and aggregation.
- `src/eval/policy_baselines.py`: shared deterministic baseline contract.
- `scripts/run_final_memory_confirmation.py`: sealed external evaluation and baselines.
- `scripts/build_final_testbed.py`: historical Phase 6 RL testbed compiler.

Generated data, checkpoints, models, reports, virtual environments, and
Graphify outputs are ignored by Git. A clean clone contains runnable code,
configuration, tests, and the small sample CSV fixture only.

## Documentation

- [Architecture](ARCHITECTURE.md)
- [Research Status](docs/RESEARCH_STATUS.md)
- [Resumable Experiments](docs/RESUMABLE_EXPERIMENTS.md)
- [Local Rank Ensemble](docs/LOCAL_RANK_ENSEMBLE.md)
- [Final Memory Confirmation](docs/FINAL_MEMORY_CONFIRMATION.md)
