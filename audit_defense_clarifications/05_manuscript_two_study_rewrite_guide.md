# Audit Defense Clarification 05: Two-Study Architecture & Manuscript Rewrite Guide

## 1. Finding / Criticism Addressed
> *"Manuscript and supplement still describe the older experiment."*

## 2. The Planned Two-Study Architecture
Rather than intermingling incomplete legacy artifacts with the clean replication, the manuscript should be cleanly split into two distinct, coherent sections:

```
+-------------------------------------------------------------------------+
| STUDY 1: The Historical Development Retrospective (551 Jobs, 2013-2024) |
| - Documents the original 551-job development sweep across 12 candidates. |
| - Documents that all 12 candidates recorded passes = false.             |
| - Best observed candidate (global_global__coverage_025) had Sharpe 1.503 |
|   but 70.3% profit concentration in China.                              |
| - Concludes that the original development failed promotion gates.        |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| STUDY 2: Registered Clean Replication & Negative Validation (23 Feat)   |
| - Eliminates unmounted legacy artifacts and 28-feature scalers.         |
| - Trains 3 Global Temporal Transformers on clean 23-feature dataset.    |
| - Evaluates matched 7-system matrix (P0-P6) across 126 cells.           |
| - Recomputes moving-block bootstrap (H2: p=0.414; H3: p=0.697).         |
| - Establishes definitive negative validation: retrieval does not beat   |
|   simple baselines (P4 Momentum: return +14.82%, Sharpe 0.475, p=0.020). |
| - Seals prospective evaluation protocol for forward out-of-sample test.  |
+-------------------------------------------------------------------------+
```

## 3. Concrete Edits for `main.tex` and `supplementary.tex`

### A. Title & Abstract
* **Title:** Keep unchanged: *"A Reproducibility and Robustness Audit of Historical Market Memory for Equity Selection: Negative Validation Across Six Markets"*
* **Abstract:** Clarify that the paper contributes a methodology-first negative validation audit:
  > *"Across a matched 126-cell backtest and panel moving-block bootstrap ($B=1,000$, $L=21$), the causal memory system ($P_0$) fails to demonstrate statistically significant outperformance over a direct no-memory comparator ($P_1$: $\Delta\text{Sharpe} = -0.147, p_{\text{Holm}} = 0.414$) or a scalar-mean baseline ($P_2$: $\Delta\text{Sharpe} = +0.046, p_{\text{Holm}} = 0.697$). Furthermore, simple 21-day momentum ($P_4$) significantly outperforms memory retrieval ($\Delta\text{Sharpe} = -0.463, p_{\text{Holm}} = 0.020$)."*

### B. Section 4: Results (H1–H4)
* **Replace Positive Alpha Claims with Negative Validation:**
  * **H1 (State Representation):** Seed stability confirmed ($\text{CKA} = 0.45 - 0.61$; $\text{MAE}_{\text{Learned}} < \text{MAE}_{\text{Raw}} < \text{MAE}_{\text{PCA}}$), but persistent ticker (58.7%) and market (66.3%) decodability limits the claim to geometric stability rather than full semantic invariance.
  * **H2 (External Memory Advantage):** Negative result ($p_{\text{Holm}} = 0.414$; fail to reject $H_0$).
  * **H3 (Distributional Conditioning):** Negative result ($p_{\text{Holm}} = 0.697$; fail to reject $H_0$).
  * **H4 (Promotion Failure & Baseline Dominance):** Supported. Both development promotion failure and momentum dominance confirm that retrieval complexity does not transfer to risk-adjusted outperformance.

### C. Figures & Tables to Update / Drop
1. **Drop Figure 6 (ADBE Case Study):** The ADBE trade was an unmounted diagnostic from an exploratory run. Dropping Figure 6 eliminates reviewer criticism regarding missing per-trade records.
2. **Mount Table 1 to Table 5:** Replace all floating numbers in the text with direct references to the 5 verified publication tables in `manuscript_tables_latex/`.
3. **Drop Non-Synchronous External Evaluation Claims:** Present 2025 out-of-sample performance as a negative validation showing out-of-sample decay, while framing forward evaluation under the pre-registered prospective protocol.

### D. Supplement Caveats
* The 56 evidence-gap caveats can be cleanly resolved:
  * Caveats regarding missing scaler parameters: **Resolved** by citing `scaler_parameters_23_features.json`.
  * Caveats regarding missing checkpoints: **Resolved** by citing the 3 PyTorch checkpoints (`global_transformer_seed_{7,17,37}.pt`).
  * Caveats regarding unverified causality: **Resolved** by citing `historical_query_level_causality_replay.csv` (6,250 verified records).
  * Caveats regarding legacy candidate metrics: **Resolved** by assigning them strictly to Study 1 (the historical development retrospective).
