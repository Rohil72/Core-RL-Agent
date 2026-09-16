# Evaluation Path Repair and Artifact Reuse Audit Report

**Date:** 2026-09-16  
**Repository:** `Rohil72/Core-RL-Agent`  
**Base Reference Revision:** `6be9ef58aaa24d096a6ee0fc35a7568542e78874`  
**Status:** COMPLETE (Awaiting Human Review — Zero Production Compute Executed)

---

## 1. Executive Summary

During review of the 6-fold production evaluation artifacts (`outputs/a30-corrected/`), anomalous trading inactivity was identified across 198 non-passive account paths in walk-forward evaluation folds 2021 through 2025.

This audit:
1. **Isolated and reproduced the root cause:** `build_security_sample_index` defaulted `evaluation_year: int = 2020`. In `scripts/launch_a30_production.py` (`prepare_fold_datasets`), the call omitted `evaluation_year=fold_year`. As a result, all sample records and sealed predictions in folds 2021..2025 were tagged with query prefix `2020_` instead of `{fold_year}_`.
2. **Identified the masking mechanism:** In Stage 3 simulation, queries looked up under `f"{fold_year}_{s}_{t}"` failed the join against `fold_eval_qids` (`2020_...`). Because `rec.exclusion_reason` was `None` (the bars and 503-day history were completely valid), the simulation logic falsely assigned `excl_reason = "MISSING_SESSION_BAR"` and score `-math.inf`. Missing-forecast checks only evaluated admitted queries, enabling 100% of the excluded queries to escape detection.
3. **Repaired the evaluation and admission path:**
   - Made `evaluation_year: Optional[int] = None` mandatory in `build_security_sample_index` (raises `ValueError` if missing, `None`, or invalid).
   - Passed `evaluation_year=fold_year` in `launch_a30_production.py`.
   - Updated simulation admission logic: valid input windows with identifier mismatches now fail loudly as `QUERY_IDENTITY_MISMATCH` and are never converted into price-bar exclusions.
   - Strengthened release audit verification to independently reconcile candidate queries, decisions, admitted queries, model expected forecasts, and evaluation dataset population per market.
4. **Audited artifact reusability:**
   - All 36 neural model checkpoints (`best_checkpoint.pt`) trained on numeric $(X, y)$ tensors are mathematically uncorrupted and assigned **`REUSE_VERIFIED`**.
   - Downstream prediction JSON files in folds 2021..2025 and all simulation accounts are assigned **`REGENERATE_DOWNSTREAM`**.
5. **Verified with focused regression tests:** 16 tests in `tests/memory_study_v2/test_query_admission_repair.py` pass 100% on CPU in 1.48s.

**Strict Boundary Compliance:** No VM/GPU jobs were launched, no models were retrained, no production runs were triggered, and no sealed results were overwritten.

---

## 2. Root-Cause Analysis and Reproduction

### 2.1 The Query Identifier Join Defect
- In `memory_study_v2/sample_index.py`:
  ```python
  def build_security_sample_index(
      security_id: str,
      bars_df: pd.DataFrame,
      venue_calendar: Optional[VenueCalendar] = None,
      evaluation_year: int = 2020,  # <-- Permissive default masked caller omissions
      required_feature_warmup: int = 252,
      required_repr_window: int = 252,
  ) -> List[SampleIndexRecord]:
  ```
- In `scripts/launch_a30_production.py` (line 493):
  ```python
  sched_val_df = sec_cal.reindex_to_schedule(val_df, (s_min, s_max))
  recs = build_security_sample_index(sec_id, sched_val_df, venue_calendar=sec_cal) # Omitted evaluation_year=fold_year
  ```
- In Stage 2 (`generate_and_seal_policy_predictions`):
  Records written to `fold_{fold_year}/policy_predictions.json` retained `rec.query_id` with `2020_` prefix, even though `fold` was 2021..2025.
- In Stage 3 (`run_continuous_portfolio_simulation`):
  ```python
  qid = f"{fold_year}_{s}_{t}"  # e.g., 2021_China_000333.SZ_2021-05-28
  rec = sec_recs_map.get(s, {}).get(t)
  is_admitted = (rec is not None and rec.input_window_valid and (fold_eval_qids is None or qid in fold_eval_qids))
  ```
  Since `fold_eval_qids` only contained `2020_...` records, `qid in fold_eval_qids` evaluated to `False`.

### 2.2 Masking of Identity Mismatch as Missing Price Data
```python
if not is_admitted:
    excl_reason = rec.exclusion_reason if (rec and rec.exclusion_reason) else ("MISSING_SESSION_BAR" if (rec and rec.input_window_valid) else "NOT_IN_INDEX")
    cov["exclusion_reasons"][excl_reason] += 1
    cand_states[s] = ("NO_ADMISSIBLE_INPUT", excl_reason, None, None, False)
    scores[s] = -math.inf
```
When `rec.input_window_valid == True` and `rec.exclusion_reason is None`:
- `excl_reason` evaluated to `"MISSING_SESSION_BAR"`.
- `scores[s]` was forced to `-math.inf`.
- No entry orders could be planned, producing zero active trading throughout 2022–2025.
- Forecast checking occurred only inside `if is_admitted:`, so zero missing forecasts were reported.

---

## 3. Market and Fold Inactivity Investigation

| Market | Fold 2020 | Fold 2021 | Folds 2022–2025 | Root Cause |
| :--- | :--- | :--- | :--- | :--- |
| **US, UK, France, Brazil** | Active trading | Trading until ~2021-03-26 (positions entered in 2020 closed at 63-session max hold); 0 new entries thereafter | 0 trades | Fold-identity join mismatch (`2020_` vs `{fold_year}_`) |
| **China** | 0 trades | 0 trades | 0 trades | **Compound Cause:**<br>1. In 2020 and early 2021: Vendor OHLCV gaps on `2019-04-29` and `2019-04-30` caused legitimate `INVALID_BAR_IN_INPUT_WINDOW` until session ordinal 2769 (`2021-05-28`).<br>2. From `2021-05-28` through 2025: China data had valid bars, but was 100% blocked by the fold-identity defect. |
| **India** | 0 trades | 0 trades | 0 trades | **Compound Cause:**<br>1. In 2020 and early 2021: Vendor OHLCV gaps on `2019-02-13` and `2019-03-29` caused legitimate `INVALID_BAR_IN_INPUT_WINDOW` until session ordinal 2788 (`2021-04-20`).<br>2. From `2021-04-20` through 2023: India data was valid, but blocked by the fold-identity defect.<br>3. In 2024–2025: Special Saturday session `2024-01-20` (DR switchover) in calendar is missing in vendor OHLCV parquets, causing `INVALID_BAR_IN_INPUT_WINDOW` for subsequent sessions. |

---

## 4. Code Modifications

1. **`memory_study_v2/sample_index.py`**:
   - `build_security_sample_index`: Removed default `evaluation_year: int = 2020`. Added strict validation requiring an integer between 1900 and 2200, raising `ValueError` otherwise.
2. **`scripts/launch_a30_production.py`**:
   - `prepare_fold_datasets`: Explicitly passed `evaluation_year=fold_year` to `build_security_sample_index`.
   - Simulation Loop: Replaced fallback logic. Valid bars failing admission now raise `RuntimeError("QUERY_IDENTITY_MISMATCH: ...")`.
   - Release Audit: Added independent reconciliation of candidate queries, decisions, admitted queries, expected forecasts, and evaluation population per market.
3. **`scripts/generate_fold_manifest.py`**:
   - Explicitly passed `evaluation_year=2020` when indexing history for fold dimension manifests.

---

## 5. Artifact Reuse Audit Matrix

Derived from the production configuration:
- 6 walk-forward evaluation folds (2020–2025)
- 2 neural backbones: `MLPAnnual` (`MLP_ANNUAL_966_64_128_1`), `TransformerAnnual` (`TRANSFORMER_42x23_WIDTH64_HEADS4_LAYERS2_FF128_LATENT128`)
- 3 random seeds: 7, 17, 37
- Total expected neural fits: $6 \times 2 \times 3 = 36$ fits.

| Artifact Category | Scope / Path | Present Bytes | Checksum Verified | Disposition | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Neural Checkpoints (MLP)** | `outputs/a30-corrected/fold_{y}/checkpoints_MLP_*_seed{s}/best_checkpoint.pt` (18 models) | 18 / 18 files present (~875 KB each) | Match `training_identity.json` & receipt | **`REUSE_VERIFIED`** | Trained on numeric $(X, y)$ tensors filtered by session date; completely independent of query-string formatting. Mathematical weights and optimizer states are uncorrupted. |
| **Neural Checkpoints (Transformer)** | `outputs/a30-corrected/fold_{y}/checkpoints_TRANSFORMER_*_seed{s}/best_checkpoint.pt` (18 models) | 18 / 18 files present (~993 KB each) | Match `training_identity.json` & receipt | **`REUSE_VERIFIED`** | Trained on numeric $(X, y)$ tensors; model weights and optimizer states are mathematically valid and uncorrupted. |
| **Ridge Models & Scalers** | Fitted per fold during Stage 2 | In-memory during pipeline execution | N/A | **`REGENERATE_DOWNSTREAM`** | Analytical closed-form linear algebra ($O(N)$ seconds on CPU); regenerates alongside prediction stage. |
| **Memory Bank Vectors** | `fold_data["bank"]` (126-session maturity vectors) | Assembled from feature parquets | Provenance intact | **`REUSE_VERIFIED`** | Stored feature embeddings and target returns are unaffected by query string keys. |
| **Policy Predictions (Fold 2020)** | `outputs/a30-corrected/fold_2020/policy_predictions.json` (553,826 records) | 137,718,318 bytes | Matches `predictions_manifest.json` | **`REUSE_VERIFIED`** | Generated with `2020_` prefix, which was correct for fold 2020. |
| **Policy Predictions (Folds 2021–2025)** | `outputs/a30-corrected/fold_{y}/policy_predictions.json` (3,512,234 records) | 5 files, ~877 MB total | Match manifest digests | **`REGENERATE_DOWNSTREAM`** | Contain `'query_id': '2020_...'` instead of `'{fold_year}_...'`. Requires re-emission with corrected fold keys. |
| **Account Ledgers** | `outputs/a30-corrected/accounts/*.json` (204 accounts) | 204 files | N/A | **`REGENERATE_DOWNSTREAM`** | Folds 2021–2025 were starved of trade entries due to query admission mismatch. |
| **Release Analysis & Contrasts** | `outputs/a30-corrected/release_analysis/` | Multiple JSON files | Matches release manifest | **`REGENERATE_DOWNSTREAM`** | Statistical contrasts reflect inactive accounts during 2021–2025; must be recomputed upon valid account replay. |

---

## 6. Staged Recovery Plan (Conditional Proposal — No Execution)

| Stage | Operations | Inputs Reused | Outputs Regenerated | Compute / Workload | Estimated Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Provenance & Local Prep** | Commit bug fix, audit report, and test evidence; verify repo integrity. | Working tree | Commit & review packet | CPU (Local) | < 2 min |
| **Stage 2: Prediction Generation** | Run model inference with corrected `evaluation_year=fold_year` using existing verified neural checkpoints. | 36 `best_checkpoint.pt` models, OHLCV parquets, memory banks | `fold_{y}/policy_predictions.json`, manifests | GPU / CPU (36 models $\times$ 6 folds inference) | 5–10 min (NVIDIA L4) |
| **Stage 3: Portfolio Replay** | Execute `run_continuous_portfolio_simulation` across all 204 accounts and 1,568 sessions with valid query joins. | `sec_info`, corrected predictions | `accounts/*.json`, `coverage_manifest.json` | CPU (Single thread / multi-process) | 3–6 min |
| **Stage 4: Statistical Inference** | Compute daily returns, primary contrasts, stationary bootstrap (10,000 draws). | Account daily returns | `primary_contrasts.json`, `replay_report.json` | CPU (NumPy vectorization) | 4–8 min |
| **Stage 5: Release Verification** | Strengthened release audit validates query population, forecast parity, checksums. | Generated artifacts | `release_manifest.json`, `pipeline_completion.json` | CPU | < 1 min |
| **Stage 6: Retraining** | **NONE REQUIRED.** Checkpoints are verified mathematically uncorrupted. | N/A | None | N/A | 0 min |

### Proposed Resource Allocation Cap
- **Maximum initial VM allocation:** **30–60 minutes** on standard VM with NVIDIA L4 GPU.
- **Spending Cap Purpose:** Profiling and executing Stages 2–5 only after human review and explicit authorization.

---

## 7. Verification Evidence

### Test Suite Execution
Command:
```bash
pytest tests/memory_study_v2/test_query_admission_repair.py tests/memory_study_v2/test_calendar_coverage_repair.py tests/memory_study_v2/test_sample_index_admission.py
```
Output:
```text
============================== 29 passed in 3.42s ==============================
```
- `tests/memory_study_v2/test_query_admission_repair.py`: 16 passed, 0 failed.
- `tests/memory_study_v2/test_calendar_coverage_repair.py`: 10 passed, 0 failed.
- `tests/memory_study_v2/test_sample_index_admission.py`: 3 passed, 0 failed.

---

## 8. Compliance Attestation

> **Explicit statement:**  
> No production rerun, retraining, paid profiling, or manuscript-result replacement was performed. This commit is awaiting review.
