# Core-RL-Agent

[![Release](https://img.shields.io/badge/release-paper--v1.0.0-blue.svg)](https://github.com/Rohil72/Core-RL-Agent/releases/tag/paper-v1.0.0)
[![Data](https://img.shields.io/badge/data-historical--memory--equity--data-green.svg)](https://github.com/Rohil72/historical-memory-equity-data)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.13-blue)](https://www.python.org/)

**Core-RL-Agent** is the official research code and evaluation framework accompanying the manuscript:
> *"A Reproducibility and Robustness Audit of Historical Market Memory for Equity Selection: Negative Validation Across Six Markets"*  
> Submitted to *Digital Finance* (Springer Nature).

---

## Quickstart: Replaying the Certified Analysis (< 15 seconds)

```bash
# Clone the repository
git clone https://github.com/Rohil72/Core-RL-Agent.git
cd Core-RL-Agent
git checkout paper-v1.0.0

# Install minimal analysis requirements
pip install numpy pandas pyarrow pytest

# Run automated invariant tests
pytest tests/test_release_contract.py tests/test_final_analysis_reconciliation.py -v
```

See [REPRODUCING.md](REPRODUCING.md) for full reproduction levels and instructions.  
See [release/experiment_contract.json](release/experiment_contract.json) for the formal mathematical and structural contract.

---

# Core RL Agent

Core RL Agent is a research codebase for causal, retrieval-grounded equity
decisions. It is not a live-trading system and no policy is currently promoted.
The active path is a patch Transformer market-state encoder, market-local
historical memory, and a deterministic continuous-rank policy that converts
three independent retrieval views into trades. The active closing study holds
the encoder and policy fixed while testing static versus growing, recency-aware,
and country-balanced raw memory. Offline RL remains only as a rejected Phase 6
comparison path.

The cycle detector remains only for earlier experiments. The final encoder uses
`patch_transformer`, disables detector targets, and sets detector action loss to
zero.

## Research Status

The sealed global/global reliability candidate failed confirmation and remains
immutable. The corrected closing memory repair did not restore broad transfer.
The active final study now asks why: stale historical evidence, source-market
imbalance, neighborhood semantics, or insufficient frozen-state signal. It does
not reopen or overwrite the failed confirmation, and no policy is promoted.

See [Research Status](docs/RESEARCH_STATUS.md) for the evidence boundary,
rejected hypotheses, promotion standard, and known limitations.

The chronological experiment ledger is in
[Research Record](docs/research_record/README.md). It records confirmed
results, rejected branches, diagnostic-only evidence, and unresolved data or
scoring defects without overwriting the phase implementation documents.

```text
daily technical and point-in-time fundamental features
  -> 252-session patch Transformer
  -> 128-dimensional latent state
  -> explicit 128-dimensional raw retrieval view
  -> static or causally growing global memory
  -> optional temporal decay and source-market balancing
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

Compile the transfer-credibility study after restoring the preserved Phase 6
and `final_memory_repair_v2` artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\run_final_memory_study.py `
  --config configs\final_transfer_credibility.yaml `
  --run-id final_transfer_credibility_v1 `
  --stage build `
  --python .venv\Scripts\python.exe
```

The stage-by-stage execution and confirmation-lock procedure is in
[Resumable Experiments](docs/RESUMABLE_EXPERIMENTS.md).

## Active Testbed

`configs/final_transfer_credibility.yaml` fixes the active comparison:

- Existing global patch Transformer outputs; no retraining.
- Explicit raw 128-dimensional retrieval views and exact C0 outcomes.
- Static, growing, two/four-year decay, and hard four-year memory variants.
- Country-balanced evidence and exact outcome-maturity admission.
- Three-seed continuous mean rank with zero required binary votes.
- ElasticNet, gradient-boosted, and PCA-kNN frozen-state decoders.
- Dependence-aware bootstrap, country jackknife, DSR, PBO, and mechanism audits.
- Markets: US, India, China, Brazil, France, and UK.
- No RL and no reliability filter.
- Development, selection, and already-observed diagnostic periods are reported
  separately. None can create a fresh confirmation claim.

`configs/final_research_testbed.yaml` and `scripts/build_final_testbed.py`
preserve the completed regional/global/offline-RL experiment for reproducibility,
but they are no longer the active execution path.

The active contract and commands are documented in
[Transfer Credibility Study](docs/FINAL_TRANSFER_CREDIBILITY_STUDY.md). The rejected reconstruction is
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
- `scripts/run_transfer_credibility_audit.py`: transfer statistics and mechanism audit.
- `src/eval/tabular_decoders.py`: conventional frozen-state decoder baselines.
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
- [Transfer Credibility Study](docs/FINAL_TRANSFER_CREDIBILITY_STUDY.md)
- [Local Rank Ensemble](docs/LOCAL_RANK_ENSEMBLE.md)
- [Final Memory Confirmation](docs/FINAL_MEMORY_CONFIRMATION.md)
