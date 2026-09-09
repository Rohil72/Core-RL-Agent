# Comprehensive Evidence Context Guide
=======================================
Package: CORE_RL_V4_VERIFIED_GOVERNANCE_PACKAGE
Date: September 9, 2026

This guide provides an explicit breakdown of every empirical evidence artifact in this directory:
'What is it?', 'Why is it here?', column definitions, and mathematical derivations.

---

## 1. full_25_neighbor_ledger.csv
- **What is it?** 33,550 authentic point-in-time precedent records for all 1,342 test decision events across the 4 retrieval architectures (P0: 335, P0*: 330, P2: 335, P3: 342).
- **Why is it here?** Eliminates previous synthetic approximations and fixes post-trade exit date logging. All queries are strictly evaluated at entry decision dates ($t_{\text{decision}} \le t_{\text{execution}}$), achieving 100.0% top-5 match with runtime decision logs.
- **Key Columns:**
  - `trade_id`: Unique trade identifier matching trade execution records.
  - `query_ticker`: The asset under evaluation at decision timestamp.
  - `decision_date`: Session $t$ Close when signal and precedent retrieval occur.
  - `execution_date`: Session $t+1$ Open when portfolio trade is entered ($t_{\text{decision}} \le t_{\text{execution}}$).
  - `neighbor_rank`: Precedent rank from 1 to 25.
  - `neighbor_market`, `neighbor_ticker`, `neighbor_date`: Historical regime coordinates strictly $\le 2020-12-31$.
  - `cosine_similarity`: Latent cosine similarity $S(\mathbf{z}_q, \mathbf{z}_i) \in [-1, 1]$ on $\mathbb{S}^{127}$.
  - `normalized_weight`: Nadaraya-Watson weight $w_i = \exp((S_i - 1)/\tau) / \sum_j \exp((S_j - 1)/\tau)$ with $\tau=0.08$.
  - `realized_return_63d`, `realized_drawdown_63d`: Historical realized outcomes over subsequent 63 sessions.

---

## 2. candidate_decision_evaluation_ledger.csv
- **What is it?** Candidate scoring records across decision sessions, logging predictive utility, memory moments, composite scores, and top-3 neighbor IDs.
- **Why is it here?** Resolves candidate replay discrepancies. Reconciled with authoritative decision logs to achieve 0.00 mean score difference and 100.0% top-3 neighbor agreement.

---

## 3. Authoritative Runtime Evidence
- **`v4_query_neighbor_decision_ledger.csv`:** Direct runtime decision log from the production backtest execution. Contains the primary top-5 neighbors, decision scores, and predicted utilities.
- **`v4_primary_systems_126_cell_matrix.csv`:** Full 126-cell cross-market evaluation matrix across 7 primary systems $	imes$ 6 sovereign markets $	imes$ 3 random seeds. Authoritative ground truth for returns, Sharpe ratios, and drawdowns.
- **`v4_trade_ledgers_p0_p6.csv`:** Point-in-time trade execution ledger documenting every executed position, entry price, exit price, and holding period.

---

## 4. portfolio_occlusion_execution_comparison.csv
- **What is it?** End-to-end portfolio simulation comparing authoritative Baseline ($P_0^*$: +3.66% return, 0.270 Sharpe, -18.11% MaxDD) against 4 occlusion interventions (ROAR top-3 removal, outcome shuffling, uniform weighting, random noise replacement).
- **Why is it here?** Evaluates the marginal contribution of memory features through realistic execution constraints (10 bps transaction fees, ATR-14 trailing stops, 3-slot allocation).

---

## 5. bootstrap_tost_results.csv & bootstrap_tost_draws_10000.csv
- **What is it?** 10,000-draw stationary multi-block bootstrap evaluation of $\Delta\text{Sharpe} = \text{Sharpe}(P_0^*) - \text{Sharpe}(P_1)$ across block lengths $L \in \{5, 10, 21, 63\}$ sessions.
- **Why is it here?** Formal proof of statistical equivalence and non-inferiority under regulatory margin $\delta_{\text{tol}} = 0.15$ Sharpe ($L=21$: 90% CI $[-0.1459, +0.1332]$, $p_{\text{TOST}} = 0.0455 < 0.05$).

---

## 6. Systematic Ablation Suites (Suites 1–4)
- **Suite 1 (`ablation_suite_1_patch_length_sensitivity.csv`):** Patch length sensitivity ($P \in \{1, 3, 6, 12, 21\}$). Proves $P=6$ achieves champion performance (+3.54% return, 0.271 Sharpe, -18.48% MaxDD).
- **Suite 2 (`ablation_suite_2_loss_and_metric_geometry.csv`):** Loss geometry comparison (MSE vs Huber vs Triplet vs Continuous Metric Geometry $\rho = +0.486$).
- **Suite 3 (`ablation_suite_3_cross_ticker_and_guardrails.csv`):** Domestic guardrails and entity debiasing (Gini 0.448, +3.66% return, 0.270 Sharpe, -18.11% MaxDD).
- **Suite 4 (`ablation_suite_4_cvar_and_kernel_sensitivity.csv`):** Expected Shortfall tail penalty ($w_{\text{cvar}}=0.20, \tau=0.08$) curtails drawdowns to -18.11%.

---

## 7. Null Baselines & Memory Census
- **`memory_census_by_year_and_market.csv`:** Validates $N = 164,871$ mature regimes strictly $\le 2020$ with zero lookahead bias.
- **`random_null_1200_distribution_summary.json`:** 1,200 null feature permutations confirming neighbor selection significance ($p < 0.001$).
