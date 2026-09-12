# Canonical Experiment Registry and Bundle Disposition Audit
**Digital Finance Forensic Governance Overhaul**  
**Date:** September 2026  
**Git Commit SHA:** `a3fb72c5b24819eafcc748765cc7078d19af4e51`  
**Data Manifest SHA-256:** `bdf79b3866b3659b1e14e41453f31d558bbc057ab3e7af7c706a97762c33d7f5`

---

## 1. Executive Summary & Completion Gate (Priority P0.1)

Under the forensic overhaul protocol, every numerical claim, table, and figure presented in the manuscript must resolve to exactly one immutable canonical artifact. All legacy, interim, and reconstructed bundles are formally audited, classified, and assigned an immutable disposition status.

| Bundle / Directory | Forensic Status | Primary Role & Disposition |
|:---|:---:|:---|
| `canonical_benchmark_outputs/` | **Canonical** | Complete 144-cell matched benchmark reproduction artifacts across all 8 systems (P0–P6) with strict 0.0% entity leakage and panel block bootstrap. |
| `paper/internal/evidence/canonical_benchmark_reproduction/` | **Canonical** | Internal authoritative mirror of clean 144-cell benchmark reproduction artifacts. |
| `FORENSIC_REMEDY_PACKAGE/` | **Canonical** | Complete forensic remedies for Points 1–5: primary P0 candidate interventions, 0.0% leakage clean P3 ledger, tail-risk CVaR discrimination data, full candidate Rank IC panel, and continuous KM RMST analysis. |
| `exports/CORE_RL_V4_VERIFIED_GOVERNANCE_PACKAGE/` | **Historical** | Retained strictly for frozen model checkpoints (`models/v4_metric_transformer_seed_*.pt`). Tabular ledgers contain reconstructed rather than raw runtime neighbors and are superseded. |
| `CORE_RL_V4_COMPACT_PACKAGE/` | **Superseded** | Deprecated interim package. Must NOT be cited or referenced in the manuscript. |
| `FINAL_SUBMISSION_PACKAGE/` | **Superseded** | Pre-audit submission bundle superseded due to audited 20.0% entity leakage in P3 and unadjusted TOST inference. OHLCV data cache retained for data integrity manifest. |

---

## 2. Canonical Experiment Mapping Matrix

Each manuscript table, figure, and empirical claim is cryptographically linked to its generating experiment, source checkpoint, and raw output artifact:

| Experiment ID | Manuscript Target | Systems / Scope | Frozen Checkpoint SHA-256 | Raw Output Artifact | Status |
|:---|:---|:---|:---|:---|:---:|
| `EXP-001-CANONICAL-144CELL-BENCHMARK` | Table 1 & Table 2 | P0, P0*, P1, P2, P3, P4, P5, P6 (144 cells) | Multi-seed (7, 17, 37) | `canonical_benchmark_outputs/canonical_144_cell_performance_matrix.csv` | **Canonical** |
| `EXP-002-FAITHFULNESS-COUNTERFACTUAL-INTERVENTION` | Section 5 & Faithfulness Table | Primary P0 & P0* (243 sessions) | Seed 7: `3355e4f1a118c2e4...` | `FORENSIC_REMEDY_PACKAGE/primary_p0_candidate_ledger.csv` | **Canonical** |
| `EXP-003-TAIL-RISK-DISCRIMINATION` | Section 4.3 & CVaR Discrimination | Primary P0 (4,126 evaluations) | Seed 7: `3355e4f1a118c2e4...` | `FORENSIC_REMEDY_PACKAGE/tail_cvar_discrimination_data.csv` | **Canonical** |
| `EXP-004-CROSS-SECTIONAL-RANK-IC` | Section 4.4 & Rank IC Analysis | Full Candidate Universe (4,126 rows) | Seed 7: `3355e4f1a118c2e4...` | `FORENSIC_REMEDY_PACKAGE/full_candidate_63d_outcome_panel.csv` | **Canonical** |
| `EXP-005-LONG-HORIZON-KM-RMST` | Section 4.2 & Survival Curve | Primary P0 (335 trades) | Seed 7: `3355e4f1a118c2e4...` | `FORENSIC_REMEDY_PACKAGE/km_survival_rmst_results.json` | **Canonical** |
| `EXP-006-CLEAN-P3-ABLATION` | Section 4.1 (Clean P3 Control) | P3 Control (342 trades, 6,840 precedents) | Raw Feature Euclidean kNN | `FORENSIC_REMEDY_PACKAGE/clean_p3_neighbor_ledger.csv` | **Canonical** |
| `EXP-007-RANDOM-POOL-BASELINE` | Section 4.1 (Distributional Null) | Uniform Random Null (1,000 draws) | Uninformed allocation null | `FORENSIC_REMEDY_PACKAGE/random_pool_baseline_draws.csv` | **Canonical** |

---

## 3. Cryptographic Governance Hashes

1. **Git Commit SHA**: `a3fb72c5b24819eafcc748765cc7078d19af4e51`
2. **Data Manifest SHA-256 (103 OHLCV parquets)**: `bdf79b3866b3659b1e14e41453f31d558bbc057ab3e7af7c706a97762c33d7f5`
3. **Model Weights Checkpoint Hashes**:
   - `Seed 7`: `3355e4f1a118c2e4c44f647bf3bcae43444a8b7e41498216c39abaf67659aba6`
   - `Seed 17`: `81a774c9794a99e6651808d603ba944d4036c0b0faabd6025b702be8f4c36213`
   - `Seed 37`: `92c6422d14fee7210ef496d00d8136299034f080b2e31904a619c3c658970d40`

---

## 4. Frozen Execution & Methodological Contract

All experiments adhere without deviation to the frozen contract:
- **Feature Dimension**: Exactly 23 causal technical indicators standardized strictly against pre-2021 historical moments per market.
- **Sovereign Universe**: 103 liquid equities across US (18), India (18), China (18), Brazil (15), France (17), UK (17).
- **Evaluation Timing**: Signal formed at session $t$ close; trade filled at session $t+1$ open.
- **Transaction Costs**: 10 bps one-way turnover (20 bps round-trip) deducted at fill execution.
- **FX Policy**: Local sovereign currency units; zero synthetic FX cross-rate conversion volatility.
- **Risk & Exit Policy**: Dynamic $2.5 \times \text{ATR}_{14}$ Chandelier trailing stop (10% minimum floor), holding period capped at 63 trading sessions.
- **Entity Leakage**: Strict 0.0% entity leakage ($\text{query\_ticker} \neq \text{neighbor\_ticker}$) across all retrieval systems.
- **Statistical Inference**: Panel-preserving moving-block bootstrap ($B=10,000, L=21$) with Benjamini-Hochberg FDR correction.
