# Audit Defense & Reviewer Clarifications Dossier

This sub-directory contains focused, technical, and econometric clarifications resolving the six primary evidence and methodology inquiries raised in the review of:

> **"A Reproducibility and Robustness Audit of Historical Market Memory for Equity Selection: Negative Validation Across Six Markets"**  
> *Target Journal:* Springer Nature — *Digital Finance*

---

## Index of Clarification Documents

1. [`01_causality_and_execution_audit.md`](file:///c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent/audit_defense_clarifications/01_causality_and_execution_audit.md)
   * **Finding:** Causality log loop distribution across all 6 global markets and strict mathematical verification of execution timing ($T_{\text{query}} < T_{\text{signal}} < T_{\text{entry}}$).
2. [`02_external_evaluation_and_india_cutoff.md`](file:///c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent/audit_defense_clarifications/02_external_evaluation_and_india_cutoff.md)
   * **Finding:** Rationale for freezing Indian equities on 31 March 2025 (historical dataset boundary), avoiding synthetic price pollution, and specification of the true out-of-sample $P_0$ policy execution contract.
3. [`03_statistical_bootstrap_and_fdr_audit.md`](file:///c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent/audit_defense_clarifications/03_statistical_bootstrap_and_fdr_audit.md)
   * **Finding:** Complete Benjamini-Hochberg False Discovery Rate (FDR) calculations, Holm-Bonferroni FWER controls, and moving-block bootstrap sample size ($B=1,000$, $L \in \{5, 21, 63\}$) justification.
4. [`04_h1_representation_reconciliation.md`](file:///c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent/audit_defense_clarifications/04_h1_representation_reconciliation.md)
   * **Finding:** Exact side-by-side reconciliation table showing 100% agreement between stored JSON diagnostics and research dossier text (Linear CKA: 0.448–0.611; MAE: Learned < Raw < PCA; decodability: 58.7% ticker, 66.3% market).
5. [`05_manuscript_two_study_rewrite_guide.md`](file:///c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent/audit_defense_clarifications/05_manuscript_two_study_rewrite_guide.md)
   * **Finding:** Concrete architectural roadmap for restructuring the manuscript into **Study 1 (Historical Retrospective Audit)** and **Study 2 (Registered Replication & Negative Validation)**, including exact LaTeX section migrations and resolution of the 56 evidence caveats.

---

## Verification & Source Artifacts
All supporting data, checkpoints, trade ledgers, and scripts are packaged in:
* Directory: [`FINAL_SUBMISSION_PACKAGE/`](file:///c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent/FINAL_SUBMISSION_PACKAGE)
* Standalone Zip: `FINAL_SUBMISSION_PACKAGE.zip` (17.85 MB)
* Master Manifest: [`FINAL_SUBMISSION_PACKAGE/MANIFEST.csv`](file:///c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent/FINAL_SUBMISSION_PACKAGE/MANIFEST.csv)
* Cryptographic Hashes: [`FINAL_SUBMISSION_PACKAGE/SHA256SUMS.txt`](file:///c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent/FINAL_SUBMISSION_PACKAGE/SHA256SUMS.txt)
