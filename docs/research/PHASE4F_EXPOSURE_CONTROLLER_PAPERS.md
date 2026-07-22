# Phase 4F Research References

This list preserves the research used to design the causal exposure controller. Phase 4F borrows general mechanisms from these papers; it does not claim to reproduce any paper's exact portfolio, universe, or empirical result.

## Risk Scaling

1. Alan Moreira and Tyler Muir, **Volatility-Managed Portfolios**, *Journal of Finance* 72(4), 2017, 1611-1644. [NBER working paper and citation](https://www.nber.org/papers/w22208), [DOI](https://doi.org/10.3386/w22208).
   - Used for: reducing exposure when lagged realized volatility rises.
   - Phase 4F constraint: long-only scaling capped at 1.0, so the controller never adds leverage.

2. Pedro Barroso and Pedro Santa-Clara, **Momentum Has Its Moments**, *Journal of Financial Economics* 116(1), 2015, 111-120. [Primary manuscript record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2041429), [DOI](https://doi.org/10.1016/j.jfineco.2014.11.010).
   - Used for: treating time-varying strategy risk as forecastable from information available before the trade.

3. Kent Daniel and Tobias J. Moskowitz, **Momentum Crashes**, *Journal of Financial Economics* 122(2), 2016, 221-247. [NBER working paper and citation](https://www.nber.org/papers/w20439), [DOI](https://doi.org/10.1016/j.jfineco.2016.07.008).
   - Used for: separating opportunity ranking from conditionally changing portfolio exposure.

## Drawdown And Execution

4. Alexei Chekhlov, Stanislav Uryasev, and Michael Zabarankin, **Drawdown Measure in Portfolio Optimization**, *International Journal of Theoretical and Applied Finance* 8(1), 2005, 13-58. [Author-hosted paper](https://www.math.columbia.edu/~chekhlov/IJTheoreticalAppliedFinance.8.1.2005.pdf), [DOI](https://doi.org/10.1142/S0219024905002767).
   - Used for: scaling risk against the portfolio underwater curve rather than reacting only to independent trade losses.
   - Phase 4F implementation: a smooth soft-to-hard drawdown budget, not a drawdown-optimized portfolio solver.

5. Zefeng Bai, Dessislava Pachamanova, Victoria Steblovskaya, and Kai Wallbaum, **Target Volatility Strategies: Optimal Rebalancing Boundary for Transaction Cost Minimization**, *Financial Markets and Portfolio Management*, 2025. [Open-access article](https://link.springer.com/article/10.1007/s11408-025-00486-5), [DOI](https://doi.org/10.1007/s11408-025-00486-5).
   - Used for: applying an explicit no-trade/rebalance threshold so small target changes do not create daily turnover.

## Confidence, Selection, And Anti-Overfitting

6. Yonatan Geifman and Ran El-Yaniv, **Selective Classification for Deep Neural Networks**, *NeurIPS 2017*. [Proceedings page](https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html), [arXiv](https://arxiv.org/abs/1705.08500).
   - Used for: the risk-coverage principle. Weak or contradictory memory evidence reduces participation instead of forcing a full-risk decision.
   - Phase 4F uses a deterministic evidence budget over frozen retrieval diagnostics; it does not train a selective neural classifier.

7. David H. Bailey and Marcos Lopez de Prado, **The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality**, *Journal of Portfolio Management* 40(5), 2014, 94-107. [Primary manuscript record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551), [DOI](https://doi.org/10.2139/ssrn.2460551).
   - Used for: requiring a multiple-testing-adjusted Sharpe probability before confirmation opens.

8. David H. Bailey, Jonathan M. Borwein, Marcos Lopez de Prado, and Qiji Jim Zhu, **The Probability of Backtest Overfitting**, *Journal of Computational Finance* 20(4), 2017, 39-69. [DOI and article record](https://doi.org/10.21314/JCF.2016.322), [SSRN record](https://ssrn.com/abstract=2326253).
   - Used for: retaining the PBO gate across the predeclared controller hypotheses.

## Research Boundary

- Transformer weights, latent geometry, memory neighbors, A2 outcome semantics, and the entry policy remain frozen.
- Only lagged portfolio returns, prior drawdown, and same-day memory diagnostics known before execution may affect exposure.
- The five controller candidates are ablations declared before evaluation, not a threshold grid.
- The 2024 test and unseen-ticker holdout remain inaccessible until every cross-market validation gate passes.
