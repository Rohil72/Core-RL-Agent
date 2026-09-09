# LaTeX Tables Manuscript Placement & Context Guide
===================================================
Package: CORE_RL_V4_VERIFIED_GOVERNANCE_PACKAGE
Date: September 9, 2026

Guide for placing the 9 publication-ready LaTeX tables into the Springer manuscript source files.

| Table File | Target Section | Primary Empirical Finding |
| :--- | :--- | :--- |
| `table_ablation_suite_1_patch_length.tex` | Section 4.3 (Model Architecture) | $P=6$ weekly macro-patch achieves champion performance (+3.54% return, 0.271 Sharpe, -18.48% MaxDD). |
| `table_ablation_suite_2_metric_geometry.tex` | Section 4.4 (Representation Geometry) | Continuous Metric Geometry maximizes rank correlation ($ho = +0.486$). |
| `table_ablation_suite_3_cross_ticker_guardrails.tex` | Section 4.5 (Retrieval Architecture) | Domestic universe guardrail controls hubness (Gini 0.448) and maximizes return (+3.66%, 0.270 Sharpe, -18.11% MaxDD). |
| `table_ablation_suite_4_cvar_governance.tex` | Section 4.6 (Tail-Risk Governance) | Expected Shortfall penalty ($w_{\text{cvar}}=0.20, \tau=0.08$) curtails drawdowns to -18.11%. |
| `table_bootstrap_tost_non_inferiority.tex` | Section 5.1 (Equivalence Proof) | TOST establishes non-inferiority at primary monthly block ($L=21$: $p=0.0455 < 0.05$). |
| `table_portfolio_occlusion_execution.tex` | Section 5.2 (Portfolio Intervention) | Quantifies marginal performance impact of precedent removal under 10 bps transaction drag. |
| `table_p0_vs_p3_clean_ablation.tex` | Section 5.3 (Architectural Progression) | Characterizes cumulative progression from single-session $k$NN ($P_3$) to annual patch Transformer ($P_1$). |
| `table_random_null_1200_distribution.tex` | Section 5.4 (Null Permutations) | Confirms precedent selection is informative over 1,200 random draws ($p < 0.001$). |
| `table_memory_bank_census.tex` | Section 3.2 (Data & Governance) | Confirms $N = 164,871$ precedents across 2000–2020 with zero lookahead bias. |
