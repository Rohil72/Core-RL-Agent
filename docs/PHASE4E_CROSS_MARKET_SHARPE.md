# Phase 4E: Cross-Market Sharpe

Phase 4E targets a pooled out-of-sample Sharpe ratio of 2.0 without changing transformer weights, retrieval geometry, or 2024 results during tuning.

## Scientific Contract

- All 14 complete Phase 4C fold/seed runs participate by default.
- Validation is performed across five fixed market groups and seeds 7, 17, and 37.
- A0, A1, and A2 use the original latent geometry. Neighbors are retrieved once per model and re-aggregated under each outcome semantic.
- The policy grid is declared in YAML before evaluation. There is no adaptive threshold search.
- Leave-one-fold-out selection measures whether the tuning procedure transfers to an unseen market group.
- The 2024 test and unseen-ticker holdout remain locked unless every promotion gate passes.
- A high Sharpe with too few trades, negative excess return, excessive drawdown, weak DSR, or excessive PBO cannot unlock confirmation.

The current universe is the complete Phase 4C equity universe. Additional asset classes or exchanges require separate frozen source-run manifests and should be added as new market groups, not merged into the tuning period after seeing results.

## Commands

Activate the repaired environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run a one-model contract check:

```powershell
python scripts/run_phase4e_cross_market_sharpe.py --stage all --run-id phase4e_smoke --max-runs 1 --smoke
```

Run production as resumable stages:

```powershell
python scripts/run_phase4e_cross_market_sharpe.py --stage score --run-id phase4e_v1
python scripts/run_phase4e_cross_market_sharpe.py --stage sweep --run-id phase4e_v1 --resume
python scripts/run_phase4e_cross_market_sharpe.py --stage select --run-id phase4e_v1 --resume
python scripts/run_phase4e_cross_market_sharpe.py --stage confirm --run-id phase4e_v1 --resume
```

The single-command equivalent is:

```powershell
python scripts/run_phase4e_cross_market_sharpe.py --stage all --run-id phase4e_v1
```

## Outputs

- `source_manifest.json`: hashes and missing-run audit.
- `evaluation_summary.csv`: semantic retrieval results across every split.
- `policy_sweep_summary.csv`: every predeclared candidate by fold and seed.
- `policy_leaderboard.csv`: pooled Sharpe, fold stability, excess return, drawdown, DSR, and PBO.
- `cross_market_oof.csv`: leave-one-fold-out selections and held-out results.
- `selected_policy.yaml`: immutable selected challenger and promotion decision.
- `selection_report.md`: readable decision report.
- `confirmation_status.json`: proof that 2024 was opened or remained sealed.
