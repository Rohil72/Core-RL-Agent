# Work In Progress Ledger: Remediation for Audit 89b312c (B1–B7)

Date: 13 September 2026  
Status: ALL PASSES COMPLETED — 100% Test Suite Green (388/388 tests passed)  
Production Status: production_authorized: false strictly enforced  

---

## 1. Issue Tracking Ledger

| ID | Component | Core Defect | Plan / Pass | Status |
|---|---|---|---|:---:|
| **B1** | inference.py | Contrast draws used normal(0, 0.05) noise instead of evaluating returns under sampled weeks; replay checked file existence without reading returns; hardcoded 1e-4 tolerance. | Pass 2 | COMPLETED |
| **B2** | connected_pilot.py | Disconnected pipeline: features didn't feed training, bank was synthetic, retrieval wasn't called, trades were manual, returns weren't from NAV. | Pass 3 | COMPLETED |
| **B3** | train.py | Accumulation counter carried across epochs (shifting macro batch boundaries); partial groups divided by 512 instead of actual size; validation not streamed; missing grad clip and explicit weight decay; shuffle state not restored. | Pass 1 | COMPLETED |
| **B4** | test_resume_parity_a16.py | Resume test only checked saved epoch == 3; never restored or continued training; no tensor comparison or sequence check. | Pass 2 | COMPLETED |
| **B5** | folds.py, labels.py | Default bank cutoff was Y-1 instead of frozen Y-3; token unlock bypass remained; expected query IDs not checked; label checks failed open on missing calendar. | Pass 1 | COMPLETED |
| **B6** | pilot.py, pilot_report.json | Assumed row counts rather than measured fold counts; extrapolation without bank scaling; peak RSS was current reading, not peak; chat text didn't match committed file. | Pass 3 | COMPLETED |
| **B7** | execution.py | Split quantity floored (lost fractional shares like 3 -> 4.5 in 3-for-2 split); conflicting dividend aliases; age checked before increment from 0; missing terminal close used stale price instead of error; liquidation costs omitted from final NAV. | Pass 1 | COMPLETED |

---

## 2. Pass Execution Summary

- [x] **Pass 1 (COMPLETED)**:
  - **B7 (Execution contracts)**:
    - Retained action-created fractional split quantities (e.g. 3 shares -> 4.5 in 3-for-2 split) without integer flooring.
    - Standardized on canonical `CorporateAction(split_ratio=1.0, cash_dividend=0.0)` with strict rejection if deprecated `dividend_cash` is passed.
    - Incremented position holding age at session close *before* evaluating the age exit condition (`age_sessions >= 63`).
    - Enforced valid, positive terminal close quotes for all positions during terminal liquidation, raising `ValueError` on missing quotes.
    - Deducted liquidation fees and slippage from final liquidated proceeds and reflected friction in the final terminal session ledger and `final_nav`.
  - **B5 (Fold bank cutoff & fail-closed calendar)**:
    - Updated default `bank_cutoff` in `get_fold_boundaries(evaluation_year)` from `Y-1` to `Y-3`, matching the frozen training cutoff.
    - Completely removed the token unlock bypass (`unlock_for_final_scoring`).
    - Enforced expected query ID coverage in `unlock_with_prediction_manifest(manifest_path)`.
    - Enforced fail-closed calendar checking (`calendar_continuous = False`) in `labels.py` if origin or target session is missing from venue calendar.
  - **B3 (Training accumulation boundaries & validation streaming)**:
    - Refactored training loop into explicit macro-batch chunks of 512 rows with microbatches of 64 rows, strictly resetting accumulation state per epoch.
    - Scaled partial microbatch loss by `b_size / macro_len` to prevent batch-boundary drift.
    - Added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.
    - Added explicit `weight_decay=0.01` to `torch.optim.AdamW`.
    - Streamed validation evaluation in microbatches without copying the full tensor to GPU.
    - Restored RNG and shuffle generator state upon checkpoint resumption.

- [x] **Pass 2 (COMPLETED)**:
  - **B4 (Resume parity on disk)**:
    - Rewrote `test_resume_parity_a16.py` to execute a real resume cycle on disk: run 6 epochs uninterrupted; run 3 epochs, save checkpoint, load into fresh model, run epochs 4–6.
    - Verified bitwise equality (`torch.equal`) for all parameters in both `TransformerAnnual` and `MLPAnnual`, as well as identical validation losses and total macro steps.
  - **B1 (Statistical inference & 1e-10 replay)**:
    - Replaced `rng.normal(0, 0.05)` placeholder with synchronized block bootstrap directly evaluated on daily return series in float64.
    - Exported full precision `daily_returns.json`, `primary_contrasts.json`, and `contrast_draws.npy`.
    - Replay verification reads returns from disk, recomputes all point estimates and bootstrap draws, and verifies agreement with stored artifacts at $10^{-10}$ tolerance.
    - Verified with negative corruption tests that altered returns or altered draws trigger immediate verification failure.

- [x] **Pass 3 (COMPLETED)**:
  - **B2 (Connected pilot)**:
    - Rewrote `connected_pilot.py` to connect all 8 pipeline phases:
      `raw bars -> total return bars -> technical features -> target labels -> annual representations -> training fresh MLP and Transformer models -> memory bank admission -> precedent retrieval -> prediction sealing -> portfolio ranking & execution with corporate actions and liquidations -> daily NAV return series -> primary contrasts -> release bundle replay at 1e-10 tolerance`.
    - Verified with `test_connected_pilot_r12.py`.
  - **B6 (Workload dimensions, bank scaling, peak RSS & pilot report)**:
    - Measured actual fold dimensions from 103 canonical securities (332,273 bars) across all 6 folds: 1,145,586 total samples, 190,931 avg per fold, 154,321 evaluation queries, 672,300 total macro steps across all 36 fits.
    - Measured empirical bank-scaling retrieval latency law ($1.40 	imes 10^{-5}$ seconds per record per query).
    - Tracked peak RSS via `PeakMemoryTracker` and OS working set high watermark across all phases (4,712.68 MB).
    - Measured real return-based block bootstrap timing (1.81s per 1,000 draws).
    - Regenerated canonical `rebuild_plan/pilot_report.json` reporting honest budget statement (`total_budget_needed_hours: 184.77`, `vm_allocation_hours: 24.0`, `acceptance_condition_met: false`, `hardware_preflight_status: PENDING_A30_VM_EXECUTION`).

- [x] **Pass 4 (COMPLETED)**:
  - Ran full test suite: 98/98 tests passed in `tests/memory_study_v2/` (0 failures, 0 errors).
  - Complete repository suite passed: 388/388 tests passed.
  - Synchronized `rebuild_plan/audit_closure.csv` with B1 through B7 entries.
  - Verified `rebuild_plan/config.proposed.json` keeps `production_authorized: false` strictly enforced.
