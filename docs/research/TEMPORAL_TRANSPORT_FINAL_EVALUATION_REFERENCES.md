# Temporal Transport Final Evaluation: Evidence and Citation Ledger

## Purpose

This ledger records why the final evaluation exists, which methodological choices are literature-grounded, and which claims remain specific to this project. It accompanies `configs/temporal_transport_final_evaluation.yaml` and `scripts/run_temporal_transport_final_evaluation.py`.

The final run does **not** train or select another model. It re-evaluates the already-frozen baseline and temporal-transport encoders after correcting the multi-market accounting protocol. The 2024 split remains the development comparison. The 2025-01-01 to 2026-03-31 split is explicitly post-selection and cannot be relabeled as untouched confirmation.

## Decision Ledger

| Decision | Empirical or methodological basis | Implemented test | Claim boundary |
|---|---|---|---|
| Freeze the two encoder variants | Repeated model selection on the same history inflates apparent performance | Exactly `baseline` and `temporal_transport`; no threshold or loss sweep | The run can compare these frozen variants, not discover a new optimum |
| Evaluate independent market books | The source backtest pooled six currencies and could omit positions on foreign-market holidays | One local-currency book per market; closed market return is zero; pooled result is an equal-weight normalized return index | The pooled curve is not a tradeable FX-converted global wealth portfolio |
| Carry last valid marks | An open holding does not become worthless when its exchange is closed | Shared backtester carries the last finite mark; equity invariants reject one-session collapses | Carry-forward is an accounting repair, not alpha generation |
| Preserve paired temporal dependence | Daily returns and holding-period strategies are serially dependent | Paired stationary bootstrap, 2,000 samples, mean block length 21 sessions | The interval measures sampling uncertainty under the fixed study; it does not remove research-selection bias |
| Include fixed cost sensitivity | Trading conclusions should not hinge on one optimistic execution assumption | 0, 10, 25, and 50 bps, declared before the final rerun | These are stylized costs, not market-specific impact models |
| Include conventional baselines | Memory performance needs economic context | Direct model head, 21-session momentum, equal-weight buy-and-hold, and 20 repeated random rankings | Baselines share execution mechanics where applicable; they are not claimed to exhaust all investable strategies |
| Keep the later period diagnostic | Its results have already influenced the research process | Reported separately and excluded from the development gate | It is evidence about observed degradation, not pristine out-of-sample confirmation |
| Require seed and market breadth | A pooled mean can hide concentration | Seed wins, market-seed win fraction, worst drawdown, detailed market ledgers | Passing the gate is supporting evidence only; promotion remains disabled |

## Representation and Architecture Sources

### 1. Patch-based time-series transformer

Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. (2023). *A Time Series Is Worth 64 Words: Long-term Forecasting with Transformers*. ICLR 2023. [OpenReview](https://openreview.net/forum?id=Jbdc0vTOcol).

Project use: supports the architectural choice to tokenize temporal patches so attention can cover longer histories efficiently. It does not establish profitability or retrieval quality in equities.

### 2. Supervised contrastive geometry

Khosla, P., et al. (2020). *Supervised Contrastive Learning*. NeurIPS 2020. [Proceedings](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html).

Project use: motivates outcome-supervised neighborhood structure rather than optimizing only a prediction head. The project uses continuous future-outcome similarity, not class labels, so its analogue loss is an adaptation rather than a reproduction of SupCon.

### 3. Alignment and uniformity

Wang, T. and Isola, P. (2020). *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*. ICML 2020, PMLR 119:9929-9939. [PMLR](https://proceedings.mlr.press/v119/wang20k.html).

Project use: motivates measuring whether similar outcomes align without allowing the latent distribution to collapse. It does not imply that better geometric metrics must improve trading utility.

### 4. Variance-covariance anti-collapse regularization

Bardes, A., Ponce, J., and LeCun, Y. (2022). *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*. ICLR 2022. [OpenReview](https://openreview.net/forum?id=xm6YD62D1Ub).

Project use: motivates the candidate's small variance/covariance regularizer. The implementation preserves per-dimension variance and penalizes off-diagonal covariance; it does not use the complete two-view VICReg training recipe.

### 5. Margin-based triplet geometry

Schroff, F., Kalenichenko, D., and Philbin, J. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering*. CVPR 2015. DOI: [10.1109/CVPR.2015.7298682](https://doi.org/10.1109/CVPR.2015.7298682).

Project use: motivates the temporal-transport hinge ordering: a cross-market, cross-period sample with similar realized outcomes should be closer than a dissimilar outcome. The project's positive/negative mining and continuous market outcomes are bespoke. The term `transport` does not mean that a formal optimal-transport objective was solved.

## Evaluation and Statistical Sources

### 6. Dependent-data bootstrap

Politis, D. N. and Romano, J. P. (1994). *The Stationary Bootstrap*. Journal of the American Statistical Association, 89(428), 1303-1313. DOI: [10.1080/01621459.1994.10476870](https://doi.org/10.1080/01621459.1994.10476870).

Project use: directly supports resampling random-length blocks rather than independent days. The implementation applies identical bootstrap indices to baseline and candidate returns, preserving paired comparisons.

### 7. Sharpe ratio dependence and uncertainty

Lo, A. W. (2002). *The Statistics of Sharpe Ratios*. Financial Analysts Journal, 58(4), 36-52. DOI: [10.2469/faj.v58.n4.2453](https://doi.org/10.2469/faj.v58.n4.2453).

Project use: supports treating annualized Sharpe estimates cautiously when returns are serially dependent. The final report therefore supplies return histories and block-bootstrap intervals instead of treating a point Sharpe as sufficient proof.

### 8. Data-snooping correction

White, H. (2000). *A Reality Check for Data Snooping*. Econometrica, 68(5), 1097-1126. DOI: [10.1111/1468-0262.00152](https://doi.org/10.1111/1468-0262.00152).

Project use: supports freezing the comparison family and documenting every evaluated alternative. The final harness does not claim to implement White's full Reality Check.

### 9. Trading-rule multiplicity and bootstrap evaluation

Sullivan, R., Timmermann, A., and White, H. (1999). *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*. Journal of Finance, 54(5), 1647-1691. DOI: [10.1111/0022-1082.00163](https://doi.org/10.1111/0022-1082.00163).

Project use: supports evaluating a declared strategy family against common baselines while retaining the history of tested variants. It does not validate this project's rules or markets.

### 10. Probability of backtest overfitting

Bailey, D. H., Borwein, J. M., Lopez de Prado, M., and Zhu, Q. J. (2015). *The Probability of Backtest Overfitting*. Journal of Computational Finance. DOI: [10.21314/JCF.2016.322](https://doi.org/10.21314/JCF.2016.322); [SSRN record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253).

Project use: supports the project's earlier PBO gates and the final refusal to reinterpret an observed split as untouched after repeated access. The final two-variant run is too small and too historically selected to manufacture a new meaningful PBO estimate.

### 11. Deflated Sharpe ratio

Bailey, D. H. and Lopez de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*. Journal of Portfolio Management, 40(5), 94-107. [SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551).

Project use: supports reporting the trial history and avoiding a standalone Sharpe claim. A defensible DSR requires a credible count and dependence structure for the full historical research family; this closing run therefore does not present a conveniently narrow DSR as if only two trials ever occurred.

### 12. Momentum baseline

Jegadeesh, N. and Titman, S. (1993). *Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency*. Journal of Finance, 48(1), 65-91. DOI: [10.1111/j.1540-6261.1993.tb04702.x](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x).

Project use: supports inclusion of a transparent price-momentum comparator. The implemented 21-session long-only cross-sectional score is deliberately simpler and shorter-horizon than the paper's canonical formation and holding portfolios.

## Claims This Final Run Can Support

1. Whether temporal-transport representation training changes retrieval and trading outcomes relative to the frozen regression baseline under corrected accounting.
2. Whether any improvement is broad across seeds and markets or concentrated in a few cohorts.
3. Whether the conclusion survives a fixed range of transaction-cost assumptions.
4. Whether paired performance differences remain directionally stable under dependent-return resampling.
5. How retrieval compares with the direct head, momentum, buy-and-hold, and repeated random ranking under the declared protocol.

## Claims This Final Run Cannot Support

1. Untouched prospective performance for 2025-2026.
2. A globally tradeable multi-currency portfolio return without FX conversion and funding rules.
3. Causal economic inference from the learned latent geometry.
4. Universal superiority across countries, market regimes, or policy families.
5. A new best configuration selected from the final results.

Those boundaries are part of the empirical result, not caveats to be removed during manuscript preparation.
