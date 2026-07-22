# Phase 4G: Memory Transfer Reliability

Phase 4G asks whether the memory system can identify, before execution, which historical analogues are likely to transfer.

## Data Contract

- Reliability fit source: causally retrieved A2 states from 2021-2022 train latents.
- Calibration source: chronological tail of the training period after a 63-session embargo.
- Validation source: frozen 2023 A2 signals.
- Seed handling: predictions and neighbours are combined into one ticker-date consensus row.
- Confirmation source: 2024 test and unseen-ticker holdout, opened only after promotion.

No transformer inference, weight update, neighbour change, or validation-derived threshold fitting occurs.

## Reliability Features

- Existing confidence, agreement, effective sample size, entropy, density, and diversity.
- Median retrieval distance, alpha interval width, alpha uncertainty, downside CVaR, and opportunity score.
- Positive-alpha market breadth and cross-sectional score dispersion.
- Cross-seed alpha dispersion, confidence dispersion, score dispersion, and neighbour-set Jaccard overlap.
- Robust train-relative state-shift score.

The classifier is fixed regularized logistic regression. A chronological calibration tail supplies isotonic calibration and coverage thresholds. A Huber model independently estimates absolute alpha error.

## Fixed Policies

- `coverage_100`: no reliability abstention.
- `coverage_075`: training-calibrated top 75% reliability coverage.
- `coverage_050`: training-calibrated top 50% reliability coverage.
- `coverage_025`: training-calibrated top 25% reliability coverage.

Realized validation coverage may differ from nominal coverage. This difference is itself evidence of distribution shift.

## Commands

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests/test_memory_reliability.py tests/test_phase4g_memory_reliability.py -q
python scripts/run_phase4g_memory_reliability.py --stage all --run-id phase4g_smoke --max-runs 1 --smoke
python scripts/run_phase4g_memory_reliability.py --stage materialize --run-id phase4g_v1
python scripts/run_phase4g_memory_reliability.py --stage build --run-id phase4g_v1 --resume
python scripts/run_phase4g_memory_reliability.py --stage evaluate --run-id phase4g_v1 --resume
python scripts/run_phase4g_memory_reliability.py --stage trade --run-id phase4g_v1 --resume
python scripts/run_phase4g_memory_reliability.py --stage select --run-id phase4g_v1 --resume
python scripts/run_phase4g_memory_reliability.py --stage confirm --run-id phase4g_v1 --resume
```

## Outputs

- `datasets/{split}/fold_*.parquet`: seed-consensus reliability datasets.
- `models/fold_*/model_audit.json`: coefficients, calibration thresholds, split dates, and shift reference.
- `predictions/{split}/fold_*.parquet`: reliability probability, predicted error, and shift score.
- `reliability_metrics.csv`: Brier skill, calibration, ordering, interval coverage, and shift diagnostics.
- `univariate_reliability.csv`: train-binned feature transfer analysis.
- `risk_coverage.csv`: outcome quality at fixed calibrated coverage.
- `coverage_backtest_summary.csv`: trading results by fold and coverage.
- `reliability_policy_leaderboard.csv`: promotion statistics.
- `cross_market_oof.csv`: leave-one-fold-out coverage selection.
- `confirmation_status.json`: explicit confirmation lock state.
