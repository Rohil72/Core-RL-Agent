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

### C1: Connected Pilot on Real Candidate Slice without Imputation or Substitute Returns
- **Data Provenance**: Sourced real market parquet files from preserved pre-2020 cache (`US_AAPL.parquet`, `US_MSFT.parquet`, 1,259 canonical sessions spanning 2013-01-02 to 2017-12-29).
- **Organic Invariants**:
  - Feature warmup: 252 bars required and satisfied before feature extraction begins.
  - Representation window: 252 bars required, slicing strictly $t-252$ to $t-1$ (origin $t$ excluded).
  - Target horizon: 63 sessions forward, organically observed from real prices.
  - Memory bank maturity: 126 sessions forward maturity strictly satisfied for all bank records before evaluation begins ($t_{\text{orig}} + 126 \le T_{\text{eval}}$).
  - Imputation policy: **Zero label imputation** (0 imputed, 0 fallback to 0.01).
- **Multi-Policy Evaluation**:
  - `MEM_SIM`: Retrieval-based policy using cosine similarity over mature bank records.
  - `MLP_BASE`: Trained MLP neural network predicting forward return scores.
  - `TRANS_BASE`: Trained Transformer neural network predicting forward return scores.
  - Each policy drives its own `PortfolioAccount` execution simulation with ranking, open fills, and stops, yielding authentic distinct daily NAV returns.
- **Un-Run Comparison Arms**:
  - Arms not executed in this restricted pilot (`KNN_PLAIN`, `MEM_RANDOM`, `HIST_PRIOR`, `RIDGE_ANNUAL`, `MLP_MIX_SR`, `TRANS_MIX_SR`, `MLP_GATE`, `TRANS_GATE`) are evaluated under `allow_reduced_arms=True`.
  - Contrasts are recorded with `status = "NOT_RUN"`, `theta = 0.0`, `ci_lower = 0.0`, `ci_upper = 0.0`, `p_value = 1.0`, **with zero substitute random noise**.
- **Manifests & Cryptographic Traceability**:
  - `input_manifest.json`: Records SHA-256 digests, row counts, security IDs, session bounds, and verification that imputed target count is 0.
  - `output_manifest.json`: Records SHA-256 digests of all generated artifacts, model checkpoints, predictions, release analysis files, and replay status.
- **Verification**: `python -m memory_study_v2.connected_pilot` runs end-to-end; verified by `tests/memory_study_v2/test_connected_pilot_r12.py`.

### C2: Calendar-Week Block Bootstrap with Weekly Sufficient Statistics & Oracle
- **Mathematical Identity (Weekly Sufficient Statistics)**:
  - For $T$ daily returns partitioned into calendar weeks $w \in \{1, \dots, W\}$, each week has sample count $N_w$, sum of returns $S_{1, w} = \sum_{t \in w} R_t$, and sum of squared returns $S_{2, w} = \sum_{t \in w} R_t^2$.
  - For any bootstrap draw of weeks, the total sample mean and variance are mathematically identical to concatenating daily returns:
    $$\sum_{t \in \text{sample}} R_t = \sum_w c_w S_{1, w}, \quad \sum_{t \in \text{sample}} R_t^2 = \sum_w c_w S_{2, w}$$
  - Evaluated via matrix multiplication $F @ S_1$ and $F @ S_2$ where $F \in \mathbb{R}^{B \times W}$ is the week frequency matrix.
  - **Memory Reduction**: Replaces 24.67 GiB tensor blowup with a ~1.5 MB frequency array ($>1,000\times$ memory reduction), executing 10,000 draws in $<0.05$ seconds with identical numerical precision ($<10^{-15}$ discrepancy).
- **Calendar-Week Non-Wrapping Partitioning**:
  - Slices trading days into non-wrapping calendar-week blocks stratified by year (`sample_calendar_week_blocks`).
- **Independent Small Numerical Oracle**:
  - Verified on 10 sessions across 2 markets and 11 arms against exact analytical expectations in `test_numerical_oracle_with_fixed_blocks`.
- **Corruption Tests**:
  - Verified that deliberate corruption of a single float in `contrast_draws.npy` or `daily_returns.json` immediately raises `ReplayVerificationError`.
- **Deterministic Replay**:
  - Persists `sampled_weeks.npy` and `bootstrap_spec.json`. Independent replay recomputes contrast estimates and draws, verifying agreement at $10^{-10}$ tolerance.

### C3: Configuration-to-Optimizer Binding
- **Strict Binding**:
  - Created `load_neural_config` in `train.py` binding `weight_decay = 0.0001` (matching `config.proposed.json:177`), `lr = 0.001`, `betas = (0.9, 0.999)`, `eps = 1e-8`.
  - Added strict assertion in `train_backbone_model` verifying optimizer parameter groups conform exactly to configuration.
- **Fail-Closed File Existence**:
  - Added explicit `FileNotFoundError` check when requested `resume_from_checkpoint` path does not exist on disk.
- **Regression Tests**: Verified by `tests/memory_study_v2/test_training_runner_r02.py`.

### C4: Terminal Liquidation Lifecycle and NAV Reconciliation
- **Call-Order Invariant**:
  - Case A (close stops evaluated first): Updates the existing terminal session row in `daily_history` so that `total_nav` and `daily_return` reflect liquidation friction and turnover.
  - Case B (terminal liquidation called directly): Computes `prev_eq` from prior session and appends a properly reconciled `DailyLedgerState` row.
- **Compound Return Reconciliation**:
  - Proves exact mathematical identity:
    $$\text{initial\_capital} \times \prod_{t=1}^T (1 + r_t) == \text{final\_nav} == \text{cash}$$
  - Guarantees zero duplicate session timestamps in ledger history.
- **Regression Tests**: Verified by 7 tests in `tests/memory_study_v2/test_execution_correctness_r05.py`.

### C5: Full-State Interruption, Mid-Epoch Recovery, and Process Boundary Resume Parity
- **Mid-Epoch Shuffling Parity**:
  - Stored `epoch_permutation` in `TrainingState`. When resuming mid-epoch (`initial_cursor > 0`), the runner reuses the exact saved permutation, guaranteeing identical downstream RNG and microbatch data delivery.
- **Macro-Boundary Interruption**:
  - Added `interrupt_at_macro_step` parameter to `train_backbone_model`.
- **Four-Way Parity Suite (`test_resume_parity_a16.py`)**:
  1. `test_transformer_resume_parity_on_disk`: Parameter and optimizer tensor equality (`exp_avg`, `exp_avg_sq`, `step`).
  2. `test_mlp_resume_parity_on_disk`: Parameter and optimizer tensor equality.
  3. `test_resume_parity_mid_epoch_macro_boundary`: Interrupted mid-epoch at macro step 5, resumed, verified bitwise equality with uninterrupted run.
  4. `test_resume_parity_fresh_process`: Interrupted at epoch 2, resumed in a completely fresh OS Python subprocess (`subprocess.run`), verified bitwise parameter and optimizer tensor equality.

### C6: Workload Projection & Honest Budget
- **Profiling with Corrected Inference**:
  - Integrated bounded-memory weekly sufficient statistics inference in `pilot.py` (1,000 draws in 0.0006s; projected 10,000 draws in $<0.01$ hours).
  - Bound AdamW `weight_decay` to `0.0001` matching configuration.
- **Traceable Fold Dimensions**:
  - Based on 103 canonical parquet files (1,145,586 total samples across 6 walk-forward folds, average 190,931 samples/fold).
- **Explicit Demarcation of Unmeasured Phases**:
  - Declared `unmeasured_phases` section in `pilot_report.json`:
    - `a30_gpu_hardware_acceleration`: UNMEASURED_LOCALLY (pending physical execution on dedicated A30 VM).
    - `cloud_persistent_storage_io`: UNMEASURED_LOCALLY (pending measurement of VM network egress to backup storage).
    - `multi_worker_parallel_retrieval`: UNMEASURED_LOCALLY (multi-core scale-out pending on 103 securities).
    - `hardware_preflight_status`: `"PENDING_A30_VM_EXECUTION"`.
  - Truthfully reports `acceptance_condition_met: false` on local profiling hardware (178.35 hours projected on local single-threaded CPU retrieval), enforcing the production block.

---

## 3. Evidence Verification Summary

| Suite | Tests | Result | Execution Time |
|---|:---:|:---:|:---:|
| `tests/memory_study_v2/` | 107 | **107 / 107 PASSED** | 13.03s |
| Complete Repository (`tests/`) | 397 | **397 / 397 PASSED** | 147.23s |
| Connected Pilot CLI (`python -m memory_study_v2.connected_pilot`) | End-to-End | **SUCCESS** | 6.5s |
| Operational Pilot CLI (`python -m memory_study_v2.pilot`) | End-to-End | **SUCCESS** | 10.5s |

**Production Guard Verification**:
- `rebuild_plan/config.proposed.json`: `"production_authorized": false` strictly maintained.
- `rebuild_plan/pilot_report.json`: `"acceptance_condition_met": false` and `"hardware_preflight_status": "PENDING_A30_VM_EXECUTION"`.
