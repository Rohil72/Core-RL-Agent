# Empirical Evidence & Schema Context Guide
Package: CORE_RL_V4_VERIFIED_GOVERNANCE_PACKAGE
Registration Date: September 2026

## 1. Primary Empirical Files
- `v4_query_neighbor_decision_ledger.csv`: Point-in-time runtime allocation logs (2,343 decisions across systems P0-P5).
- `v4_primary_systems_126_cell_matrix.csv`: Comprehensive 126-cell matrix (6 sovereign markets x 3 seeds x 7 systems).
- `v4_trade_ledgers_p0_p6.csv`: Chronological trade execution ledger with exact entry, peak, exit, holding days, and fees.
- `full_25_neighbor_ledger.csv`: 33,550 historical precedents (25 per trade) with strict t_decision <= t_execution and dates <= 2020.
- `candidate_decision_evaluation_ledger.csv`: 75,720 cross-sectional candidate scores across all decision sessions.

## 2. Research-Grade Interventions & Robustness
- `faithfulness_static_decision_matrix.csv`: Static decision estimand metrics (Rank rho, tau, Top-1 Hit%, Top-3 Overlap%).
- `faithfulness_dynamic_portfolio_matrix.csv`: Dynamic execution estimand under realistic frictions (fees, ATR chandelier stops).
- `faithfulness_stochastic_replications.csv`: Raw 3,600 stochastic replication draws for bundle swap and random replacement.
- `kaplan_meier_survival_with_greenwood_bands.csv`: Event-level survival curves with log-log Greenwood 95% confidence bands.
- `holding_survival_summary.json`: RMST restricted mean survival times, cluster bootstrap SEs, and calendar truncation audit.
- `algorithmic_worked_decisions.json`: 3 objectively selected illustrative decision cards with complete score decomposition.
- `worked_decision_path_data.parquet`: Daily observed OHLCV trajectories for queries and precedents.
- `temporally_matched_retrieval_ladder_90_cells.csv`: 90-cell sequence retrieval ladder across 5 representation rungs.
- `bootstrap_tost_draws_10000.csv`: 10,000 multi-block bootstrap TOST draws evaluating equivalence bounds.
