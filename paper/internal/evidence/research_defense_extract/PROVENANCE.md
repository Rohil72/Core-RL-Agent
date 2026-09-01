# Research Defense Extract — Provenance and Audit Note

**Document:** `paper/internal/evidence/research_defense_extract/PROVENANCE.md`  
**Prepared For:** Editorial Review, Committee Proctoring, and Manuscript Fact-Checking  
**Target Path:** `paper/internal/evidence/research_defense_extract/`  

---

## 1. Commit Provenance & Git Lineage

This extract bridges the historical testbed execution with the formal reconstruction audit:

| Commit Hash | Commit Subject & Date | Role in Research Defense | File Integrity Match |
|---|---|---|---|
| **`23922607a8d45c47c198fde609f0f046440231f7`** | `lossless continuation changes` (24 July 2026) | **Historical Principal Baseline.** Source commit under which testbed `phase6_a30_final_v1` completed training. | **128 of 132 files** match exact SHA-256 (4 non-matching were helper inspection scripts). |
| **`54e3aa9e7d9c66bc2e98faef446d3e8e2e28a50b`** | `fix: migrate spot runtime and runner atomically` (31 August 2026) | **Runtime Bridge.** Consolidated durable runner migrations and AMP recovery. | Clean working tree. |
| **`37c22591b435d84e481a0a57aa47ce9e13fa120a`** | `feat(eval): add causality audit, matched primary baselines, and research defense suite` (01 Sept 2026) | **Reconstruction & Defense Baseline.** Implements 11-target lineage audit, P0–P6 matched baselines, moving-block bootstrap, and universe ledger. | Current audited commit. |

---

## 2. Separation of Historical Outputs vs. Reconstruction Outputs

To maintain absolute scientific integrity, outputs in this extract are classified into two strictly separated categories:

### A. Original Historical Experiment Outputs (Archival Ground Truth)
* **Run Root:** `reports/final_testbed/phase6_a30_final_v1/`
* **Artifacts:**
  * `selection.json`: Authoritative promotion gate outcome (`status = rejected`, `selected = none`, `best = global_global__coverage_025`).
  * `build_summary.json`: 551 jobs completed (7 data, 61 pilot, 483 full, 0 confirmation).
  * `experiment_manifest.yaml`: Exact 551-job launch DAG and resource specifications.
  * `leaderboard.csv`: Complete 12-candidate sweep metrics (all 12 failed gates: `passes = false`).
  * `market_results.csv`: 6-market performance, Brier scores, and realized coverage table.
  * `confirmation_report.md`: External temporal evaluation report (2025–2026: Return -6.02%, Sharpe -0.376).
  * `models/*.pt`: All 21 original trained Patch Transformer neural weights (Global & Regional $\times$ Seeds 7, 17, 37).
* **Modification Status:** **UNMODIFIED.** Read-only extraction directly from the original VM testbed pack.

### B. Newly Produced Reconstruction Outputs (Audit & Baselines)
* **Run Root:** `paper/internal/evidence/research_defense_extract/` & `exports/research_defense_bundle/`
* **Artifacts:**
  * `target_lineage_table.json`: 11-target horizon mapping.
  * `target_252_isolation_proof.json`: Mathematical isolation certificate (`Violations: 0`).
  * `universe_retention_ledger.json`: 108 requested $\to$ 103 available $\to$ 5 excluded ledger.
  * `primary_systems_p0_p6.json`: Matched primary comparison matrix (P0–P6).
  * `statistical_significance_tests.json`: 21-session moving-block bootstrap ($p_{\text{Holm}}$ and $q_{\text{FDR}}$).
  * `representation_h1_diagnostics.json`: Linear CKA (0.9939), kNN overlap (0.7995), PCA controls.
* **Creation Date & Command:** Generated on **01 Sept 2026** via:
  ```bash
  python3 scripts/run_reconstruction_v1.py --stage all --device cuda
  python3 scripts/extract_research_defense_package.py --run-root reports/reconstruction_v1 --profile full
  ```
* **Reporting Requirement:** Reported as newly evaluated audit baselines and causal proofs, not historical retrospective claims.

---

## 3. Reconciliation of P0 Sharpe (1.084) vs. Historical Leaderboard Sharpe (1.503)

Reviewers noted an apparent discrepancy between P0's Sharpe ratio of `1.084` and the original best-candidate Sharpe of `1.503`. This is fully explained by portfolio aggregation:

1. **Original Historical Sweep (`1.503026`):**
   * Represents the **6-Market Cross-Market Dynamically Weighted Pooled Sharpe** for candidate `global_global__coverage_025` over the 2024 development period.
   * Driven by strong performance in China (+85.22% return, Sharpe 2.216, Calmar 8.144) which contributed **70.29% of positive profit**.
2. **Standardized Single-Market Reference P0 (`1.084`):**
   * Represents the **US Market Standalone Benchmark** for the identical global-encoder/global-memory model at 25% nominal coverage over 2024.
   * US Standalone Return: **+22.61%**, Sharpe: **1.084**, Sortino: **1.153**, Max Drawdown: **-17.16%**, Win Rate: **70.0%** across 70 trades.
3. **Equivalence Join:** In `market_results.csv`, row 1 (`US`) lists Sharpe exactly as **`1.084`**, and the pooled summary row lists **`1.503`**. Both figures are byte-identical and describe the exact same underlying run at market-level vs. pooled-level.

---

## 4. Integrity and Non-Modification Certificate

* **Raw Data Integrity:** Original price matrices and splits were not altered.
* **Checkpoints:** The 21 `.pt` checkpoints are original binaries generated during `phase6_a30_final_v1` on NVIDIA A30. Their SHA-256 digests are catalogued in `REPRODUCIBILITY_CHECKPOINT_INVENTORY.csv`.
* **Configurations:** The contract `configs/reconstruction_v1.yaml` explicitly matches the historical parameter space with tightened causal boundaries (`strict_cutoff_contract: true`).
