# Phase 4F: Causal Exposure Controller

Phase 4F tests whether portfolio risk conversion, rather than representation or retrieval, is the remaining bottleneck. It consumes frozen Phase 4E `A2_base` signals and never retrains or modifies the transformer.

## Decision Path

```text
Frozen A2 memory evidence
        +
Lagged portfolio returns and drawdown
        |
        v
CausalExposureController
        |
        v
Target gross exposure in [0, 1]
        |
        v
Existing long-only policy and next-bar execution
```

The controller calculates three auditable scalars:

- **Volatility scalar:** `min(1, target_volatility / max(lagged_volatility, floor))`.
- **Drawdown scalar:** 1 above the soft limit, then linearly decreases toward a configured floor at the hard limit.
- **Evidence scalar:** a weighted quality score from median retrieval confidence, agreement, effective sample size, and positive alpha breadth.

The combined candidate uses the minimum scalar. This makes the binding risk reason interpretable and avoids multiplying several imperfect confidence estimates into near-zero exposure.

All portfolio-risk inputs exclude the current execution bar. Memory diagnostics come from the previous signal bar, matching the existing next-bar execution contract. Exposure reductions use a configurable rebalance boundary to limit transaction churn.

## Predeclared Ablations

- `F0_a2_static`: frozen A2 policy parameters with no controller, evaluated under corrected causal execution timing.
- `F1_volatility`: volatility targeting only.
- `F2_drawdown`: drawdown response only.
- `F3_evidence`: memory-quality coverage control only.
- `F4_combined`: minimum of all three risk budgets.

No adaptive threshold search is performed. Selection uses all complete Phase 4E validation runs, leave-one-fold-out market selection, DSR, PBO, excess return, fold stability, trade count, and drawdown gates. Confirmation cannot run from smoke output.

## Commands

Activate the existing environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run focused tests:

```powershell
python -m pytest tests/test_exposure_controller.py tests/test_phase4f_exposure_controller.py tests/test_market_memory_backtester.py -q
```

Run a one-model smoke contract:

```powershell
python scripts/run_phase4f_exposure_controller.py --stage all --run-id phase4f_smoke --max-runs 1 --smoke
```

Run production as resumable stages:

```powershell
python scripts/run_phase4f_exposure_controller.py --stage audit --run-id phase4f_v1
python scripts/run_phase4f_exposure_controller.py --stage sweep --run-id phase4f_v1 --resume
python scripts/run_phase4f_exposure_controller.py --stage select --run-id phase4f_v1 --resume
python scripts/run_phase4f_exposure_controller.py --stage confirm --run-id phase4f_v1 --resume
```

The confirmation stage materializes frozen A2 test/holdout signals only when validation promotion succeeds.

## Outputs

- `source_manifest.json`: immutable hashes of every validation signal cache.
- `controller_sweep_summary.csv`: run-level performance and controller diagnostics.
- `controller_leaderboard.csv`: pooled and cross-market promotion statistics.
- `cross_market_oof.csv`: held-out market selection results.
- `selected_controller.yaml`: resolved controller and lock state.
- `selection_report.md`: readable validation decision.
- `confirmation_status.json`: explicit proof that confirmation stayed sealed or executed.
- `policies/.../exposure_decisions.csv`: daily target exposure and binding risk reason.
