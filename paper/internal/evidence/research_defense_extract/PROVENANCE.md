# Research Defense Extract — Provenance, Universe Alignment, and Audit Note

**Document:** `paper/internal/evidence/research_defense_extract/PROVENANCE.md`  
**Prepared For:** Editorial Review, Committee Proctoring, and Manuscript Fact-Checking  
**Target Directory:** `paper/internal/evidence/research_defense_extract/`  

---

## 1. Commit Provenance & Checkpoint Lineage

| Commit Hash | Commit Subject & Date | Role in Research Defense | Build Integrity & File Match |
|---|---|---|---|
| **`23922607a8d45c47c198fde609f0f046440231f7`** | `lossless continuation changes` (24 July 2026) | **Historical Principal Baseline.** Source commit under which testbed `phase6_a30_final_v1` was built. All 21 original `.pt` model checkpoints were trained under this commit. | **128 of 132 files** match exact SHA-256 (4 non-matching were helper inspection scripts). |
| **`54e3aa9e7d9c66bc2e98faef446d3e8e2e28a50b`** | `fix: migrate spot runtime and runner atomically` (31 August 2026) | **Runtime Bridge.** Consolidated durable runner migrations and AMP recovery. | Clean working tree. |
| **`37c22591b435d84e481a0a57aa47ce9e13fa120a`** | `feat(eval): add causality audit, matched primary baselines, and research defense suite` (01 Sept 2026) | **Reconstruction & Defense Baseline.** Formalized 11-target lineage audit, P0–P6 matched baselines, and moving-block bootstrap engine. | Current audited commit. |

---

## 2. Hardened Universe Alignment (100% Match to Historical `data_audit.csv`)

The universe definition in `src/data/universe_ledger.py` and `configs/reconstruction_v1.yaml` has been strictly synchronized with the historical testbed (`reports/final_testbed/phase6_a30_final_v1/data_audit.csv`):

* **United States (18 available):** `AAPL`, `ADBE`, `AMD`, `AMGN`, `AMZN`, `AVGO`, `CRM`, `GOOGL`, `INTU`, `ISRG`, `LMT`, `META`, `MSFT`, `NOC`, `NVDA`, `ORCL`, `REGN`, `V`
* **India (18 available):** `ASIANPAINT.NS`, `BAJFINANCE.NS`, `BHARTIARTL.NS`, `EICHERMOT.NS`, `HCLTECH.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`, `INFY.NS`, `LT.NS`, `M&M.NS`, `MARUTI.NS`, `PIDILITIND.NS`, `RELIANCE.NS`, `SUNPHARMA.NS`, `TCS.NS`, `TECHM.NS`, `TITAN.NS`, `ULTRACEMCO.NS`
* **China (18 available):** `000333.SZ`, `000725.SZ`, `000858.SZ`, `002230.SZ`, `002241.SZ`, `002415.SZ`, `002475.SZ`, `002594.SZ`, `300059.SZ`, `600036.SS`, `600196.SS`, `600276.SS`, `600309.SS`, `600519.SS`, `600887.SS`, `601012.SS`, `601318.SS`, `601888.SS`
* **Brazil (15 available, 3 missing):**
  * *Available:* `B3SA3.SA`, `EQTL3.SA`, `FLRY3.SA`, `ITUB4.SA`, `KLBN11.SA`, `LREN3.SA`, `MGLU3.SA`, `PETR4.SA`, `RADL3.SA`, `RAIL3.SA`, `RENT3.SA`, `SUZB3.SA`, `TOTS3.SA`, `VALE3.SA`, `WEGE3.SA`
  * *Excluded (Data unavailable):* `CIEL3.SA`, `EMBR3.SA`, `JBSS3.SA`
* **France (17 available, 1 missing):**
  * *Available:* `AI.PA`, `AIR.PA`, `CAP.PA`, `DG.PA`, `DIM.PA`, `DSY.PA`, `EL.PA`, `LR.PA`, `MC.PA`, `ML.PA`, `OR.PA`, `RI.PA`, `RMS.PA`, `SAF.PA`, `SU.PA`, `TEP.PA`, `WLN.PA`
  * *Excluded (Data unavailable):* `STM.PA`
* **United Kingdom (17 available, 1 missing):**
  * *Available:* `AUTO.L`, `AZN.L`, `BA.L`, `CPG.L`, `CRDA.L`, `DGE.L`, `EXPN.L`, `HLMA.L`, `JD.L`, `LSEG.L`, `OCDO.L`, `PRU.L`, `REL.L`, `RMV.L`, `RTO.L`, `SGE.L`, `SPX.L`
  * *Excluded (Data unavailable):* `AHT.L`
* **Total:** 108 requested $\to$ 103 available $\to$ 5 excluded.

---

## 3. Strict Runtime Causal Enforcement (Hardened Implementation)

1. **Split-Boundary Label Purging (`src/data/sequence_dataset.py`):**
   * Training split samples enforce that the label horizon ($\le 252\text{ trading sessions}$) cannot exceed the split cutoff date ($\le \text{2020-12-31}$). Samples extending past the split boundary are actively purged during sample generation.
2. **Outcome Availability & Temporal Separation (`src/memory/retrieval.py`):**
   * `require_outcome_availability: bool = True` is enforced.
   * Nearest-neighbour search requires $T_{\text{available}} \le T_{\text{query}}$ in UTC.
   * Same-ticker exclusion (`same_ticker_mode: "exclude"`) and 21-session temporal separation ($\Delta t \ge 21$) are enforced prior to distance ranking.
3. **252-Session Target Isolation Certificate (`target_252_isolation_proof.json`):**
   * Mathematical verification proves `future_max_return_252` is strictly absent from memory schemas, reliability classification, scoring formulas, and exit conditions.

---

## 4. Evaluation Window and India Historical Boundary

* **Historical Baseline Window:** The primary multi-market development sweep evaluated the 2024 calendar year.
* **India External Evaluation Endpoint:** As documented in the original study records, the historical external evaluation ledger for India terminates in March 2025. The manuscript retains this documented historical boundary rather than projecting synthetic continuous data.

---

## 5. Separation of Historical Artifacts vs. Reconstruction Pipeline

* **Historical Checkpoints (July 2026):** All 21 `.pt` neural network weights in `reports/final_testbed/phase6_a30_final_v1/models/` are original historical binaries created under commit `239226...`.
* **Historical Selection Record:** `reports/final_testbed/phase6_a30_final_v1/selection.json` and `leaderboard.csv` document the authoritative rejected status (all 12 candidate configurations failed the promotion gates).
* **Reconstruction Pipeline (September 2026):** The P0–P6 matched baseline suite and moving-block bootstrap are evaluated as modern diagnostic verification of the causal memory mechanism.
