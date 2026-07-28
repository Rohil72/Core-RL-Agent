# Core RL Agent

Core RL Agent is a research codebase for causal, retrieval-grounded equity
decisions. It is not a live-trading system and no policy is currently promoted.
The active path is a patch Transformer market-state encoder, market-local
historical memory, and a deterministic continuous-rank policy that converts
three independent retrieval views into trades. The final bounded study compares
raw and adapter retrieval, C0 and rally outcomes, ticker-balanced evidence, and
multi-scale causal memory. Offline RL remains only as a rejected Phase 6
comparison path.

The cycle detector remains only for earlier experiments. The final encoder uses
`patch_transformer`, disables detector targets, and sets detector action loss to
zero.

## Research Status

The sealed global/global reliability candidate failed confirmation and remains
immutable. The subsequent local-rank reconstruction also failed to transfer
consistently and exposed an embedding-selection ambiguity. The active final
memory study reuses its frozen artifacts to isolate raw versus adapter memory,
then tests richer rally evidence and multi-scale agreement. It does not reopen
or overwrite the failed confirmation, and no policy is promoted.

See [Research Status](docs/RESEARCH_STATUS.md) for the evidence boundary,
rejected hypotheses, promotion standard, and known limitations.

```text
daily technical and point-in-time fundamental features
  -> 252-session patch Transformer
  -> 128-dimensional latent state
  -> explicit raw or 32-dimensional adapter retrieval view
  -> static or market-local growing causal memory
  -> robust rally evidence and optional multi-scale agreement
  -> continuous three-seed mean-rank policy
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

Compile the final memory study after restoring the ignored
`local_rank_ensemble_v1` artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\run_final_memory_study.py `
  --config configs\final_memory_study.yaml `
  --run-id final_memory_study_v1 `
  --stage build `
  --python .venv\Scripts\python.exe
```

The stage-by-stage execution and confirmation-lock procedure is in
[Resumable Experiments](docs/RESUMABLE_EXPERIMENTS.md).

## Active Testbed

`configs/final_memory_study.yaml` fixes the final bounded comparison:

- Existing regional patch Transformer and adapter outputs; no retraining.
- Explicit raw 128-dimensional and adapter 32-dimensional retrieval views.
- Exact C0, rally-path, ticker-balanced, and multi-scale memory variants.
- Static and growing causal memory that admits outcomes only after maturity.
- Three-seed continuous mean rank with zero required binary votes.
- Deterministic top-k trading policy and declared standard baselines.
- Markets: US, India, China, Brazil, France, and UK.
- No RL and no reliability filter.
- Development, selection, and already-observed diagnostic periods are reported
  separately. None can create a fresh confirmation claim.

`configs/final_research_testbed.yaml` and `scripts/build_final_testbed.py`
preserve the completed regional/global/offline-RL experiment for reproducibility,
but they are no longer the active execution path.

The active contract and commands are documented in
[Final Memory Study](docs/FINAL_MEMORY_STUDY.md). The rejected reconstruction is
documented in [Local Rank Ensemble](docs/LOCAL_RANK_ENSEMBLE.md), and the failed
sealed procedure remains documented in
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
- `scripts/run_final_memory_study.py`: active memory-only DAG and aggregation.
- `scripts/run_local_rank_ensemble.py`: rejected reconstruction infrastructure.
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
- [Final Memory Study](docs/FINAL_MEMORY_STUDY.md)
- [Local Rank Ensemble](docs/LOCAL_RANK_ENSEMBLE.md)
- [Final Memory Confirmation](docs/FINAL_MEMORY_CONFIRMATION.md)
