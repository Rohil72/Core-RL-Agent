# Audit Defense Clarification 04: H1 Representation Reconciliation

## 1. Finding / Criticism Addressed
> *"H1 dossier values disagree with the JSON."*

## 2. Origin of Discrepancy
* **Legacy Prototype Metric:** Early unconstrained prototypes evaluated on synthetic feature projections produced an inflated Linear CKA of $\approx 0.99$.
* **Empirical Measured Reality:** When the registered 23-feature Global Temporal Transformer encoders were trained across seeds 7, 17, and 37 and evaluated against matched testbed sequences, the actual Linear CKA values ranged from **`0.448` to `0.611`**.
* The older dossier text erroneously retained the legacy $0.99$ placeholder.

## 3. Side-by-Side Reconciliation Table

The table below reconciles `RESEARCH_LEVEL_EVIDENCE_DOSSIER.md` with `representation_h1_diagnostics.json` in [`FINAL_SUBMISSION_PACKAGE/`](file:///c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent/FINAL_SUBMISSION_PACKAGE):

| Diagnostic Metric | Stored JSON Value | Synchronized Dossier Value | Status | Interpretation |
|---|:---:|:---:|:---:|---|
| **Linear CKA (Seed 7 vs. 17)** | `0.4481` | `0.4481` | **RECONCILED (100% Match)** | Moderate cross-initialization geometry stability |
| **Linear CKA (Seed 7 vs. 37)** | `0.6107` | `0.6107` | **RECONCILED (100% Match)** | Moderate cross-initialization geometry stability |
| **Linear CKA (Seed 17 vs. 37)** | `0.4434` | `0.4434` | **RECONCILED (100% Match)** | Pairwise initialization stability |
| **$k\text{NN}$ Jaccard Overlap ($k=25$)** | `0.2493 - 0.2832` | `0.2493 - 0.2832` | **RECONCILED (100% Match)** | Modest neighborhood overlap (vs. random chance 0.0097) |
| **Learned Neighbour Outcome MAE** | `0.1554` | `0.1554` | **RECONCILED (100% Match)** | Outperforms raw and PCA controls |
| **Raw $k\text{NN}$ Outcome MAE Control** | `0.1626` | `0.1626` | **RECONCILED (100% Match)** | Baseline control |
| **14-d PCA Outcome MAE Control** | `0.1643` | `0.1643` | **RECONCILED (100% Match)** | Linear dimension reduction control |
| **Nuisance Ticker Decodability** | `58.69%` | `58.69%` | **RECONCILED (100% Match)** | High ticker decodability (vs. chance 0.97%) |
| **Nuisance Market Decodability** | `66.28%` | `66.28%` | **RECONCILED (100% Match)** | High market decodability (vs. chance 16.67%) |

## 4. Scientific Implication for the Manuscript
* The empirical representation diagnostics support **bounded seed stability** and confirm the theoretical error ordering ($\text{MAE}_{\text{Learned}} < \text{MAE}_{\text{Raw}} < \text{MAE}_{\text{PCA}}$).
* However, because linear probes can decode ticker identity with 58.7% accuracy and market identity with 66.3% accuracy, the representation **does not achieve semantic invariance**.
* In the revised manuscript, hypothesis $H_1$ is strictly characterized as establishing **partial geometry stability with persistent entity entanglement**, not full market-invariant abstraction.
