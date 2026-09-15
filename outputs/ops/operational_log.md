# Operational Log: Core-RL-Agent Production Execution

## 1. Environment & Hardware Baseline
- **Timestamp (UTC)**: 2026-09-14T18:50:22Z
- **Commit SHA**: `c0f895f64b388553c56137b1951b128ecfbc43bb`
- **Working Tree**: Clean (no tracked files modified)
- **Python Executable**: `/root/miniconda3/envs/py3.10/bin/python3` (Python 3.10.20)
- **PyTorch / CUDA**: PyTorch 2.11.0+cu130, CUDA Runtime 13.0, Driver 595.58.03
- **GPU Device**: NVIDIA L4 (24GB VRAM, Compute Capability 8.9)
- **Host RAM**: 1,133.5 GB Total | 1,047.2 GB Available
- **Disk (`/home`)**: 100 GB Total | 99 GB Available (2% utilized)
- **VM Allocation Deadline**: Unknown from OS environment (querying operator)

## 2. Preflight Gate Verification Receipts
- **Gate 1 (Process / Session Isolation)**: PASSED. No prior production process or `core-rl` tmux session active.
- **Gate 2 (Commit & Tracked Integrity)**: PASSED. Checkout strictly resolves to `c0f895f64b388553c56137b1951b128ecfbc43bb`.
- **Gate 3 (CUDA Hardware & Kernel Probe)**: PASSED. Live 1024x1024 GEMM probe succeeded on GPU in 148.28 ms. Device is NVIDIA L4 (24GB).
- **Gate 4 (Parquet Integrity)**: PASSED. 103/103 parquet files verified against SHA-256 digests in `rebuild_plan/fold_dimensions_manifest.json` with zero mismatches.
- **Gate 5 (Sample Manifests & Venue Calendars)**: PASSED. 6 fold sample manifests present (2020–2025). All 6 market venue calendars (BRAZIL, CHINA, FRANCE, INDIA, UK, US) resolve non-inferred (`is_inferred=False`) from deterministic session fixtures.
- **Gate 6 (Repository Guard)**: PASSED. `production_authorized` remains `False` in `rebuild_plan/config.proposed.json`.
- **Gate 7 (Check-Only Execution)**: PASSED. `python3 scripts/launch_a30_production.py --check-only` generated signed receipt `rebuild_plan/a30_production_out/a30_launch_receipt.json` with status `PREFLIGHT_PASS`.

## 3. Operational Timeline
- [2026-09-14T18:50:22Z] Preflight validation completed successfully.
- [2026-09-14T18:53:35Z] Prepared production runner script `outputs/ops/run-production.sh`.
- [2026-09-14T19:20:00Z] Implemented bounded float64 roundoff policy (`memory_study_v2/ohlc_validation.py`, max 4 ULPs via `np.nextafter`). Verified 8/8 policy tests pass.
- [2026-09-14T19:30:00Z] Completed 103-security diagnostic scan: 44 securities valid, 24 roundoff-only (276 rows normalized), 35 with material vendor discrepancies (74 rows kept rejected and audited to `outputs/ops/material_discrepancies.csv`).
- [2026-09-14T19:47:00Z] Configured `validate_raw_bars` to reject material vendor rows from valid bars so downstream calendar marks sessions as MISSING without study shrinkage.
- [2026-09-14T20:01:00Z] Aligned `volume_change_1` on zero-volume previous days to 0.0 matching canonical `.replace([np.inf, -np.inf], 0.0).fillna(0.0)` policy across replication scripts.
- [2026-09-14T20:04:38Z] Dispatched authorized full production run across all 6 walk-forward folds (2020–2025) in tmux session `core-rl`.
- [2026-09-14T20:08:15Z] Milestone reached: **Fold 2020 COMPLETED** (6/6 neural training jobs finished with verified checkpoint SHA-256 digests and predictions: MLP seeds 7, 17, 37; Transformer seeds 7, 17, 37).
- [2026-09-14T21:57:00Z] Milestone reached: **Fold 2021 COMPLETED** (6/6 neural training jobs finished).
- [2026-09-14T23:15:00Z] Milestone reached: **Fold 2022 COMPLETED** (6/6 neural training jobs finished).
- [2026-09-15T01:18:00Z] Milestone reached: **Fold 2023 COMPLETED** (6/6 neural training jobs finished).
- [2026-09-15T03:07:00Z] Milestone reached: **Fold 2024 COMPLETED** (6/6 neural training jobs finished).
- [2026-09-15T05:02:00Z] Milestone reached: **Fold 2025 COMPLETED** (6/6 neural training jobs finished). Stage 1 100% complete (36/36 jobs).
- [2026-09-15T05:04:00Z] Stage 2 sealed predictions generated for all 6 folds (over 3.8 million prediction records).
- [2026-09-15T05:04:28Z] Stage 3 encountered missing quote at terminal calendar session 2025-12-31 for held position `China_000858.SZ` (the OHLCV parquet dataset ends on 2025-12-30 across all markets, while calendar fixtures listed 2025-12-31).
- [2026-09-15T06:54:50Z] Implemented Section 8.3 mark-to-market terminal liquidation quote fallback: if a held position lacks a quote on the terminal calendar date, terminal liquidation falls back to `pos.last_valid_price` with an audit notice.
- [2026-09-15T07:00:52Z] Verified full test suite passes (30/30 tests pass: `test_a30_launch.py` 23/23, `test_execution_correctness_r05.py` 7/7).
- [2026-09-15T07:01:05Z] Re-dispatched full production pipeline in tmux `core-rl`. Fast resumption successfully verified and reused all 36 trained backbone checkpoints from Stage 1 and all 6 sealed prediction manifests from Stage 2.
- [2026-09-15T07:19:40Z] **Stage 3 Completed**: Continuous portfolio simulation simulated all 204 continuous account paths across all 6 years (1,567 union sessions). Terminal liquidations reconciled exactly with cash balances.
- [2026-09-15T07:19:45Z] **Stage 4 Completed**: Evaluated primary contrasts (P1–P8) with 100 bootstrap draws and exported analysis bundle to `release_analysis/`.
- [2026-09-15T07:19:48Z] **Stage 5 Completed**: Independent replay verification passed (`REPLAY_VERIFIED`, tolerance 1e-10).
- [2026-09-15T07:19:54Z] Pipeline finished with exit code 0. Emitted `a30_launch_receipt.json` with status `PRODUCTION_SUCCESS`.

## 4. Verification Summary
- **Overall Execution Receipt**: `outputs/a30-c0f895f/a30_launch_receipt.json` (`status: PRODUCTION_SUCCESS`, `study_scope: FULL_PRODUCTION`, `accounts_simulated: 204`, `union_sessions: 1567`)
- **Stage 1 Completion**: 36/36 neural training jobs completed across 6 folds (`fold_2020` to `fold_2025`) with sealed checkpoints (`best_checkpoint.pt`) and cryptographic digests.
- **Stage 2 Completion**: 6/6 `fold_summary.json` files and `predictions_manifest.json` sealing 3,895,784 predictions.
- **Stage 3 Completion**: All 204 continuous accounts simulated across 6 global markets (Brazil, China, France, India, UK, US).
- **Stage 4 Completion**: 8 primary contrasts evaluated with block bootstrap inference.
- **Stage 5 Completion**: `release_analysis/replay_report.json` verified with zero discrepancy (`REPLAY_VERIFIED`).
- **Release Manifest**: `outputs/a30-c0f895f/release_manifest.json` emitted with status `RELEASE_VERIFIED`.
- **Repository Safety Guard**: `production_authorized: False` intact in `rebuild_plan/config.proposed.json`.
