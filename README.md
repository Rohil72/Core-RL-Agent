# Core RL Agent

Core RL Agent is a research codebase for causal, retrieval-grounded equity
decisions. It is not a live-trading system and no policy is currently promoted.
The active path is a patch Transformer market-state encoder, a historical
market-memory layer, and a deterministic confidence-aware policy that converts
retrieved evidence into trades. Offline RL remains available only as a rejected
Phase 6 comparison path.

The cycle detector remains only for earlier experiments. The final encoder uses
`patch_transformer`, disables detector targets, and sets detector action loss to
zero.

## Research Status

Development results support the global encoder with global internal and external
memory as the final candidate architecture, but they do not support a promoted
performance claim. Offline RL and regional internal-memory adaptation were
rejected. The candidate and its 25% reliability-coverage threshold are now
frozen; further work is baseline comparison and untouched external validation,
not more development-set tuning.

See [Research Status](docs/RESEARCH_STATUS.md) for the evidence boundary,
rejected hypotheses, promotion standard, and known limitations.

```text
daily technical and point-in-time fundamental features
  -> 252-session patch Transformer
  -> 128-dimensional latent state
  -> causal market memory and analogue evidence
  -> deterministic consensus and reliability policy
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

Compile the locked final-memory DAG after restoring the ignored Phase 6 source
artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\run_phase6_frozen_memory_sweep.py `
  --config configs\final_memory_policy.yaml `
  --run-id final_memory_policy_v1 `
  --stage build `
  --python .venv\Scripts\python.exe
```

The stage-by-stage execution and confirmation-lock procedure is in
[Resumable Experiments](docs/RESUMABLE_EXPERIMENTS.md).

## Final Testbed

`configs/final_memory_policy.yaml` fixes the current candidate:

- One shared global patch Transformer and global internal memory.
- Global historical external memory with causal retrieval.
- Three-seed consensus and confidence-aware deterministic trading policy.
- Reliability threshold locked to the 25% development candidate.
- Markets: US, India, China, Brazil, France, and UK.
- No RL, no encoder retraining, and no further threshold search.
- Existing promotion gates remain unchanged; the candidate is not promoted.

`configs/final_research_testbed.yaml` and `scripts/build_final_testbed.py`
preserve the completed regional/global/offline-RL experiment for reproducibility,
but they are no longer the canonical execution path.

The sealed 2025-2026 Q1 baseline and robustness procedure is documented in
[Final Memory Confirmation](docs/FINAL_MEMORY_CONFIRMATION.md). It hashes the
candidate, source, data, and development evidence before running and never
reopens architecture or threshold selection.

## Repository Map

- `src/models/patch_transformer_model.py`: active encoder.
- `src/losses/outcome_geometry.py`: regression and optional geometry losses.
- `src/memory/`: causal retrieval, evidence aggregation, and confidence.
- `src/decision/`: counterfactual decision outcomes.
- `src/policy/`: deterministic policy plus historical offline-RL comparisons.
- `src/backtest/`: memory-policy backtesting.
- `src/trainers/train_cycle_model.py`: encoder training and resumable state.
- `src/orchestration/`: immutable durable experiment runner.
- `scripts/run_phase6_frozen_memory_sweep.py`: canonical locked-memory evaluator.
- `scripts/run_final_memory_confirmation.py`: sealed external evaluation and baselines.
- `scripts/build_final_testbed.py`: historical Phase 6 RL testbed compiler.

Generated data, checkpoints, models, reports, virtual environments, and
Graphify outputs are ignored by Git. A clean clone contains runnable code,
configuration, tests, and the small sample CSV fixture only.

## Documentation

- [Architecture](ARCHITECTURE.md)
- [Research Status](docs/RESEARCH_STATUS.md)
- [Resumable Experiments](docs/RESUMABLE_EXPERIMENTS.md)
- [Final Memory Confirmation](docs/FINAL_MEMORY_CONFIRMATION.md)
