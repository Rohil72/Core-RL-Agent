# Forensic Audit & Empirical Remedy Package
**Package**: `FORENSIC_REMEDY_PACKAGE`  
**Distribution State**: Uncompressed, Self-Contained Directory  
**Date**: September 2026  

---

## 1. Executive Purpose & Governance Scope

This package directly addresses the forensic audit findings across the 4-stage empirical chain:
$$\text{Eligible Evidence} \longrightarrow \text{Used Evidence} \longrightarrow \text{Informative Evidence} \longrightarrow \text{Economic Consequence}$$

### Scope & Evidentiary Boundaries:
1. **Faithfulness Scoping**: Evidentiary faithfulness interventions establish that decisions **depend counterfactually on retrieved historical evidence** (e.g., occluding precedents alters allocations and portfolio trajectories). **Faithfulness does NOT establish that the evidence is predictive, correct, or alpha-generating.**
2. **Strict Rhetorical Hygiene**: All promotional, marketing, and speculative rhetoric has been discarded:
   - Discarded terms: *"risk governor"*, *"interpretability tax"*, *"unprecedented"*, *"regulatory compliance"*, and unverified *"clean-P3 superiority"*.
   - Retained terms: Strictly empirical descriptions (e.g., *retrospective tail constraint*, *Bayesian shrinkage / anchoring drag*, *endogenous duration without holding floor*).

---

## 2. Directory Manifest & Immediate Priority Deliverables

| Priority | Filename | Description |
|---|---|---|
| **Priority 1** | [`generate_points_1_to_5.py`](generate_points_1_to_5.py) | Master executable script regenerating all Points 1–5 metrics directly from authoritative sources. |
| **Priority 2** | [`full_candidate_63d_outcome_panel.csv`](full_candidate_63d_outcome_panel.csv) | Full cross-sectional candidate panel (4,126 rows across 243 active decision sessions) with realized 63d returns and drawdowns for true cross-sectional rank IC. |
| **Priority 3** | [`random_pool_baseline_generator.py`](random_pool_baseline_generator.py)<br>[`random_pool_baseline_draws.csv`](random_pool_baseline_draws.csv) | Uniform random precedent generator and baseline draws strictly drawn from archive ($\le 2020$) as negative control. |
| **Priority 4** | [`clean_p3_cross_ticker_rerun.py`](clean_p3_cross_ticker_rerun.py)<br>[`clean_p3_neighbor_ledger.csv`](clean_p3_neighbor_ledger.csv) | Cross-ticker P3 rerun excluding all 1,710 same-ticker leaked rows (0.0% entity leakage), re-weighted across 20 clean cross-ticker neighbours. |
| **Priority 5** | [`tail_cvar_discrimination.py`](tail_cvar_discrimination.py)<br>[`tail_cvar_discrimination_data.csv`](tail_cvar_discrimination_data.csv) | Source data and Mann-Whitney U test proving precedent downside CVaR flags forward left-tail drawdowns ($2.18\times$ risk ratio). |
| **Priority 6** | [`primary_p0_candidate_ledger.csv`](primary_p0_candidate_ledger.csv)<br>[`generate_primary_p0_interventions.py`](generate_primary_p0_interventions.py) | Authoritative candidate evaluation ledger for Primary P0 (4,126 rows, 243 sessions) and counterfactual intervention generator. |
| **Priority 7** | [`generate_km_rmst_survival.py`](generate_km_rmst_survival.py)<br>[`km_survival_rmst_results.json`](km_survival_rmst_results.json) | Continuous Kaplan-Meier survival estimator ($S(0)=1.0$), proving $\text{RMST}_{63} = 43.04$ sessions and CI $[40.21, 46.22]$. |

---

## 3. Core Forensic Discoveries Resolved

1. **P0 vs P0* Attribution**: The previously cited $13.9\%$ Top-1 and $35.9\%$ Top-3 change rates apply to exploratory $P_0^*$ (Domestic Guardrail). For Primary $P_0$ (Global Learned Memory), the true change rates are **$16.46\%$ Top-1** and **$34.16\%$ Top-3** (Spearman $\rho = 0.9648$, $5.26\%$ sign overrides).
2. **P0* Rank/Weight Anomaly Identified**: In $P_0^*$, ranks 1–5 were uniformly set to $0.0400$ ($20.0\%$ sum), while ranks 6–10 carried descending weights starting at $0.0640$, causing the top-5 largest weights to total $31.95\%$. In Primary $P_0$, weights decrease monotonically with rank (ranked top-5 = largest top-5 = $42.48\%$).
3. **P3 Entity Leakage Quantified**: Raw-feature $k\text{NN}$ ($P_3$) contained exactly 5 same-ticker precedents per trade across all 342 trades ($1,710 / 8,550$ rows, $20.0\%$). Purged in `clean_p3_neighbor_ledger.csv`.
4. **True Cross-Sectional Rank IC**: Evaluating Rank IC across all candidate evaluations per session yields mean $-0.0592$ for score and $-0.0587$ for memory $\mu$ (mitigating an even more negative direct utility of $-0.0887$).
5. **Continuous RMST Origin**: Correcting discrete summation to continuous trapezoidal integration shifts $\text{RMST}_{63}$ from $42.04$ to **$43.04$ sessions** ($SE = 1.50$), shifting the $95\%$ bootstrap confidence interval from $[39.21, 45.22]$ to **$[40.21, 46.22]$**.
