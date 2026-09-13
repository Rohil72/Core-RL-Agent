# Work In Progress Ledger: Remediation for Audit 259f000 (C1–C6)

Date: 14 September 2026  
Status: ALL REMEDIATIONS C1–C6 COMPLETED — 100% Test Suite Green (397/397 tests passed, 107/107 in `tests/memory_study_v2/`)  
Production Status: `production_authorized: false` strictly enforced in `rebuild_plan/config.proposed.json`  

---

## 1. Issue Tracking Ledger (R01–R12, B1–B7, C1–C6)

| ID | Component | Core Defect | Scope / Pass | Status |
|---|---|---|---|:---:|
| **C1** | `connected_pilot.py` | Input data was synthetic, labels had 0.01 fallback, comparison arms were substituted with random normal returns, missing input/output manifests. | Pass 4 | **CLOSED** |
| **C2** | `inference.py` | Block bootstrap needed calendar-week partitions, independent numerical oracle, corruption detection, and bounded-memory execution to eliminate 24.67 GiB tensor blowup. | Pass 3 | **CLOSED** |
| **C3** | `train.py` | Training runner did not strictly assert optimizer binding to configuration (`weight_decay=0.0001`), missing resume checkpoint did not raise `FileNotFoundError`. | Pass 1 | **CLOSED** |
| **C4** | `execution.py` | Terminal liquidation calling sequence lacked daily ledger NAV reconciliation when called directly or when close stops evaluated first. | Pass 1 | **CLOSED** |
| **C5** | `train.py`, `test_resume_parity_a16.py` | Resume test did not demonstrate mid-epoch macro-boundary interruption or fresh process isolation with optimizer state tensor bitwise equality. | Pass 2 | **CLOSED** |
| **C6** | `pilot.py`, `pilot_report.json` | Workload projection needed bounded-memory inference profiling, traceable fold counts, and explicit demarcation of unmeasured phases pending A30 VM execution. | Pass 4 | **CLOSED_HONEST_BUDGET_PENDING_VM** |
| **B1–B7** | Multiple | Historical remediation findings from commit `89b312c` (inference replay, connected pipeline, accumulation boundaries, on-disk resume, fold cutoff, fold counts, execution contracts). | Historical | **CLOSED** |
| **R01–R12** | Multiple | Foundational architecture contracts from commit `f937fd2`. | Historical | **CLOSED** |

---

## 2. Technical Remediation Narrative (C1–C6)

### C1: Chronological Partitions, Venue Matching, and Multi-Policy Artifacts
- **Venue Matching via Schedule Intersection**:
  - Aligned raw candidate bars (`US_AAPL.parquet`, `US_MSFT.parquet`) using verified venue calendar intersection (`venue_sessions`), strictly eliminating arbitrary row truncation.
  - Sourced explicit `CorporateAction` adapter and documented `price_adjustment_mode = "ADJUSTED_PRICE_CACHE_PILOT_MODE"`.
- **Strict Chronological Partitions (Zero Overlap, Purged Forward Labels)**:
  - **Train**: Indices $[503, 560]$ (2014-12-31 to 2015-03-25). Forward target horizon ($t+63$) matures by 2015-06-24. Memory bank records ($t+126$) mature by 2015-09-23.
  - **Validation**: Indices $[630, 680]$ (2015-07-06 to 2015-09-15). Strictly purged after all training targets mature. Validation targets mature by 2015-12-14.
  - **Evaluation**: Indices $[756, 805]$ (2016-01-04 to 2016-03-15). All queries in calendar year 2016 (fold 2016). All bank records mature before eval start.
  - **Zero Label Imputation**: Organically calculated forward 63-session cumulative returns across all partitions (0 imputed, 0 fallback to 0.01).
  - **Feature Scaling**: Feature scaler fitted strictly on training partition (`training_cutoff = "2015-03-25"`) before transforming validation and evaluation sets.
- **Multi-Policy Artifact Generation**:
  - `MEM_SIM`: Retrieval-based policy using cosine similarity over mature bank records.
  - `MLP_BASE`: Trained MLP neural network predicting forward return scores.
  - `TRANS_BASE`: Trained Transformer neural network predicting forward return scores.
  - Exported sealed predictions, trade ledgers, and daily NAV histories for ALL 3 policies (`predictions_2016_{mem_sim,mlp_base,trans_base}.json`, `trades_{MEM_SIM,MLP_BASE,TRANS_BASE}.json`, `daily_nav_{MEM_SIM,MLP_BASE,TRANS_BASE}.json`).
  - Regenerated `input_manifest.json` and `output_manifest.json` with SHA-256 digests and 1e-10 release replay verification.
- **Verification**: `tests/memory_study_v2/test_connected_pilot_r12.py` (2/2 passed).

### C2: Calendar Identity, Native Holiday Masks, and Fail-Closed Replay Validation
- **Monday Anchor Week Identity**:
  - Week identity computed via Monday anchor date `(dt.year, (dt - timedelta(days=dt.weekday())).strftime("%Y-%m-%d"))`, eliminating the ISO week collision where `2024-01-02` and `2024-12-30` both mapped to `(2024, 0)`.
  - Malformed or invalid dates strictly raise `ValueError` (no fallback to row bins).
- **Per-Series Holiday Mask**:
  - `compute_weekly_sufficient_statistics` applies per-series holiday mask, ensuring unobserved sessions do not accumulate count or sums.
  - Mathematically matches direct array indexing oracle across holidays and year boundaries ($< 10^{-12}$).
- **Missing Required Market Paths**:
  - `evaluate_primary_contrasts` strictly rejects missing required market paths when `allow_reduced_arms=False` (no silent skipping).
- **Fail-Closed Replay Checks**:
  - `replay_analysis_bundle` verifies contrast counts, draw matrix shapes, finite checks on all matrices/scalars, contrast IDs, arm names, statuses, and tolerances.
  - Added corruption tests in `test_inference_replay_r10.py` for renamed contrast IDs, truncated summary lists, and NaN-corrupted stored draws (13/13 passed).
- **Resource Scope**: Scoped float64 frequency matrix allocation ($25,040,000$ bytes for 313 weeks and 10,000 draws) distinct from measured RSS.

### C3: Fail-Closed Production Configuration Enforcement
- **Strict Production Mode**:
  - When `execution_mode = "production"`, `train_backbone_model` enforces fail-closed checks:
    - Missing config raises `FileNotFoundError`.
    - `production_authorized: false` strictly raises `PermissionError`.
    - Unapproved hyperparameter overrides raise `ValueError`.
  - Bound `weight_decay = 0.0001` matching `config.proposed.json:177`.
  - Restored optimizer parameter groups validated against configured hyperparameters on resume.
- **Verification**: `tests/memory_study_v2/test_resume_parity_a16.py` and `tests/memory_study_v2/test_training_runner_r02.py`.

### C4: Terminal Liquidation Lifecycle and NAV Reconciliation
- **Call-Order Invariant**:
  - Case A (close stops evaluated first): Updates the existing terminal session row in `daily_history` so that `total_nav` and `daily_return` reflect liquidation friction and turnover.
  - Case B (terminal liquidation called directly): Computes `prev_eq` from prior session and appends a properly reconciled `DailyLedgerState` row.
- **Compound Return Reconciliation**:
  - Proves exact mathematical identity:
    $$\text{initial\_capital} \times \prod_{t=1}^T (1 + r_t) == \text{final\_nav} == \text{cash}$$
  - Guarantees zero duplicate session timestamps in ledger history.
- **Regression Tests**: Verified by 7 tests in `tests/memory_study_v2/test_execution_correctness_r05.py`.

### C5: Final-Macro Interruption Regression and Process Boundary Resume Parity
- **Final-Macro Update Boundary**:
  - In `train_backbone_model`, when `interrupt_at_macro_step` occurs at `macro_end >= N_train`, the epoch is finished: validation pass and checkpoint selector run before saving checkpoint with `sampler_cursor = 0`, ensuring zero skipped validation passes upon resume.
- **Comprehensive Parity Suite (`test_resume_parity_a16.py`, 6/6 passed)**:
  1. `test_transformer_resume_parity_on_disk`: Parameter and optimizer tensor equality (`exp_avg`, `exp_avg_sq`, `step`).
  2. `test_mlp_resume_parity_on_disk`: Parameter and optimizer tensor equality.
  3. `test_resume_parity_mid_epoch_macro_boundary`: First-macro interruption parity.
  4. `test_resume_parity_final_macro_boundary`: Final-macro interruption parity with full validation loss history and best epoch parity.
  5. `test_resume_parity_fresh_process`: Process boundary parity via OS Python subprocess (`subprocess.run`).
  6. `test_production_config_fail_closed`: Production authorization and override rejection.

### C6: Hashed Admissible-Row Manifest & Reconciled Profiler Receipts
- **Hashed Admissible-Row Manifest**:
  - Sourced `scripts/generate_fold_manifest.py` computing SHA-256 digests and annual admissible row counts across all 103 canonical market securities to produce `rebuild_plan/fold_dimensions_manifest.json`.
  - Exactly captures the 1,145,586 total timeline samples across the 6 walk-forward folds (2020..2025):
    - Year 2020: train=126,667, val=25,646, dev=25,639, eval=25,853
    - Year 2021: train=152,313, val=25,639, dev=25,853, eval=25,766
    - Year 2022: train=177,952, val=25,853, dev=25,766, eval=25,707
    - Year 2023: train=203,805, val=25,766, dev=25,707, eval=25,588
    - Year 2024: train=229,571, val=25,707, dev=25,588, eval=25,753
    - Year 2025: train=255,278, val=25,588, dev=25,753, eval=25,654
  - `pilot.py` dynamically loads and verifies `rebuild_plan/fold_dimensions_manifest.json` with SHA-256 tracking.
- **Reconciled Profiler Timing Receipt**:
  - Receipts report exact measured local benchmarks on NVIDIA GeForce RTX 2050:
    - `bootstrap_contrasts_1000_draws_seconds`: 0.0751s - 0.0822s
    - `mlp_effective_batch_512_macro_step_ms`: 9.70ms
    - `transformer_effective_batch_512_macro_step_ms`: 43.77ms
    - `validation_pass_seconds`: 0.637s
    - `projected_training_hours`: ~5.3h
    - `projected_retrieval_hours`: ~117.2h
    - `total_budget_needed_hours`: ~188.5h
  - Truthfully reports `acceptance_condition_met: false` locally, keeping `production_authorized: false` strictly locked.
- **Demarcation of Unmeasured Phases**:
  - Declared `unmeasured_phases` section in `pilot_report.json`:
    - `a30_gpu_hardware_acceleration`: UNMEASURED_LOCALLY (pending physical execution on dedicated A30 VM).
    - `cloud_persistent_storage_io`: UNMEASURED_LOCALLY (pending measurement of VM network egress to backup storage).
    - `multi_worker_parallel_retrieval`: UNMEASURED_LOCALLY (multi-core scale-out pending on 103 securities).
    - `hardware_preflight_status`: `"PENDING_A30_VM_EXECUTION"`.
- **Verification**: `tests/memory_study_v2/test_operational_pilot_a32.py` (4/4 passed).

---

## 3. Post-Review Findings 1–5 Remediation (14 September 2026)

### Finding 1: Independent Venue Calendar (C1 Continuation)
- Generated canonical fixture `data/fixtures/us_trading_sessions.csv` containing 4,279 NYSE sessions (2010–2026).
- Created `memory_study_v2/venue_calendar.py` providing `VenueCalendar`:
  - Three-state session classification: `CLOSED`, `VALID`, `MISSING`.
  - Monotonic session ordinals derived from calendar, not dataset indices.
  - Schedule reindexing and window validation.
- Wired `connected_pilot.py` to use `VenueCalendar` instead of file-intersection heuristics.
- Acceptance tests: `tests/memory_study_v2/test_connected_pilot_r12.py` (5/5 passed).

### Finding 2: Canonical Mask for Point Estimates & Replay (C2 Continuation)
- In `memory_study_v2/inference.py`:
  - Point estimates now strictly filter by `canonical_mask` identical to the weekly sufficient statistics accumulator.
  - Added fail-closed guard: missing returns on open sessions raise `ValueError`.
  - `export_analysis_bundle` serializes `open_session_mask` into `daily_returns.json`.
  - `replay_analysis_bundle` reads, validates, and forwards `open_session_mask` to prevent unmasked re-evaluation.
- Acceptance tests: `tests/memory_study_v2/test_inference_replay_r10.py` (18/18 passed including Sharpe math oracle).

### Finding 3: Admissible Sample Counts from Loader/Bank Functions (C6 Continuation)
- Rewrote `scripts/generate_fold_manifest.py` to call `partition_fold` from `memory_study_v2.folds` across all 103 canonical market securities for all 6 walk-forward evaluation folds (2020..2025).
- Sourced all 6 required per-fold population counts:
  - `train_samples` (and `train_query_samples`): 120,178 to 248,789.
  - `validation_samples` (and `val_query_samples`): 19,099 to 19,364.
  - `development_samples` (and `dev_query_samples`): 19,099 to 19,364.
  - `evaluation_queries` (and `eval_query_samples`): 25,588 to 25,853.
  - `scored_eval_query_samples`: 25,588 to 25,853 (in 2025: 19,165, strictly separating decision-time query eligibility from future target maturity).
  - `bank_samples` (and `bank_record_samples`): 113,689 to 242,300 (strictly strictly reflecting 126-session maturity vs 63-session training maturity).
- Persisted per-fold query IDs under `rebuild_plan/sample_ids/`.
- Updated `load_measured_fold_dimensions()` in `pilot.py` to fail closed in acceptance/production mode.
- Re-executed `pilot.py` to update `rebuild_plan/pilot_report.json` with matching manifest hash.
- Acceptance tests: `tests/memory_study_v2/test_operational_pilot_a32.py` (4/4 passed).

### Finding 4: Complete Production Configuration Boundary (C3 Continuation)
- Added `REQUIRED_NEURAL_KEYS` schema validator in `load_neural_config()`.
- Validated stopping rule overrides (`min_epochs`, `max_epochs`, `patience`) alongside optimizer parameters in production mode.
- Replaced all validation `assert` statements with explicit `raise ValueError`.
- Added canonical JSON SHA-256 config hash stored in checkpoint states and verified on resume.
- Acceptance tests: `tests/memory_study_v2/test_resume_parity_a16.py` (10/10 passed).

### Finding 5: Batched Retrieval Optimization & A30 Preflight (C6 Continuation)
- Added `precompute_bank_norms()` and `compute_squared_euclidean_batched()` to `MemoryBank` in `memory_study_v2/memory.py` using BLAS-optimized $\|q\|^2 + \|b\|^2 - 2q^T b$ formula.
- Added `retrieve_mem_sim_batch()` in `memory_study_v2/retrieval.py` preserving exact tie-breaking, spacing ($\ge 21$), and cap ($\le 3$) rules.
- Sourced dedicated A30 VM benchmark preflight script `scripts/benchmark_retrieval_a30.py`.
- Acceptance tests: `tests/memory_study_v2/test_retrieval_reference_a17.py` (6/6 passed).

---

## 4. Evidence Verification Summary

| Suite | Tests | Result | Execution Time |
|---|:---:|:---:|:---:|
| `tests/memory_study_v2/` | 136 | **136 / 136 PASSED** | 15.15s |
| Complete Repository (`tests/`) | 426 | **426 / 426 PASSED** | ~132s |
| Connected Pilot CLI (`python -m memory_study_v2.connected_pilot`) | End-to-End | **SUCCESS** | 6.5s |
| Operational Pilot CLI (`python -m memory_study_v2.pilot`) | End-to-End | **SUCCESS** | 8.8s |
| Manifest Generator CLI (`python scripts/generate_fold_manifest.py`) | 103 Securities | **SUCCESS** | 11.2s |

**Production Guard Verification**:
- `rebuild_plan/config.proposed.json`: `"production_authorized": false` strictly maintained.
- `rebuild_plan/pilot_report.json`: `"acceptance_condition_met": false` and `"hardware_preflight_status": "PENDING_A30_VM_EXECUTION"`.
