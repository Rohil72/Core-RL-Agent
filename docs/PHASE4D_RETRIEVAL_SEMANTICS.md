# Phase 4D: Retrieval Semantics

Phase 4D keeps every transformer checkpoint frozen. It tests whether relative outcomes, a bounded retrieval metric, calibrated OOD rejection, and signal-level seed consensus make historical analogues more useful for trading.

The default production scope uses the previously identified stable fold 04 seeds 7, 17, and 37. The audit still verifies all 14 complete Phase 4C sources. Expand `experiment.active_folds` only after this bounded ladder produces a promotable result.

## Experiment Ladder

| Variant | Retrieval geometry | Outcome semantics | Score |
|---|---|---|---|
| A0 | Original latent geometry | Absolute return | Legacy evidence score |
| A1 | Original latent geometry | Universe-relative alpha | Alpha lower-confidence-bound score |
| A2 | Original latent geometry | Sector/universe blended alpha | Alpha lower-confidence-bound score |
| A3 | Diagonal frozen-latent metric | Blended alpha | Alpha lower-confidence-bound score |
| A4 | Low-rank frozen-latent metric | Blended alpha | Alpha lower-confidence-bound score |

The diagonal and low-rank adapters train only on exported latent vectors and realized outcomes through 2021, with 2022 used for early stopping. Transformer modules and checkpoints are never imported by the runner.

## Run Sequence

Use a new run ID for a production experiment. Staged commands after the first command require `--resume`.

```powershell
python scripts/run_phase4d_retrieval_semantics.py --stage audit --run-id phase4d_v1
python scripts/run_phase4d_retrieval_semantics.py --stage fit-metrics --run-id phase4d_v1 --resume
python scripts/run_phase4d_retrieval_semantics.py --stage ablate --run-id phase4d_v1 --resume
python scripts/run_phase4d_retrieval_semantics.py --stage select --run-id phase4d_v1 --resume
python scripts/run_phase4d_retrieval_semantics.py --stage diagnostic-2024 --run-id phase4d_v1 --resume
```

The equivalent single command is:

```powershell
python scripts/run_phase4d_retrieval_semantics.py --stage all --run-id phase4d_v1
```

Before production, exercise the contract on one frozen run:

```powershell
python scripts/run_phase4d_retrieval_semantics.py --stage fit-metrics --run-id phase4d_smoke --max-runs 1 --smoke
python scripts/run_phase4d_retrieval_semantics.py --stage ablate --run-id phase4d_smoke --max-runs 1 --smoke --resume
python scripts/run_phase4d_retrieval_semantics.py --stage select --run-id phase4d_smoke --max-runs 1 --smoke --resume
```

Smoke runs use 256 query rows and are marked `smoke_only`; they are not promotion evidence.

## Promotion Contract

Selection uses validation data only. A candidate must clear fold and seed win rates, drawdown regression, memory NDCG gain, a paired stationary-bootstrap lower bound, PBO, and deflated-Sharpe probability. If no candidate clears every gate, `selected_stack.yaml` explicitly falls back to A0.

Only the selected stack is evaluated on 2024 test and unseen-ticker holdout data. Fold 04 seeds 7, 17, and 37 are combined at the signal level with a two-of-three vote; incompatible latent coordinates are never averaged.

Outputs include source hashes, prepared outcomes, fitted adapters, distance calibrations, per-run signals/neighbors/trades/decisions/equity, `evaluation_summary.csv`, `ablation_ladder.csv`, and `selected_stack.yaml`.
