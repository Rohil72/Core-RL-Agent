# Phase 4G Research References

Phase 4G tests whether frozen market-memory diagnostics identify transferable states before a trade. These references define the calibration, selective-risk, dataset-shift, and distribution-shift principles used by the harness.

1. Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger, **On Calibration of Modern Neural Networks**, ICML 2017. [PMLR paper](https://proceedings.mlr.press/v70/guo17a.html).
   - Used for: separating ranking quality from probability calibration and measuring reliability diagrams, Brier score, and calibration error.

2. Yonatan Geifman and Ran El-Yaniv, **Selective Classification for Deep Neural Networks**, NeurIPS 2017. [Proceedings paper](https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html).
   - Used for: fixed risk-coverage evaluation where the system abstains from low-reliability decisions.

3. Stephan Rabanser, Stephan Guennemann, and Zachary C. Lipton, **Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift**, NeurIPS 2019. [Proceedings paper](https://proceedings.neurips.cc/paper/2019/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html).
   - Used for: reporting whether validation-state diagnostics are distinguishable from training-state diagnostics.

4. Masashi Sugiyama, Matthias Krauledat, and Klaus-Robert Mueller, **Covariate Shift Adaptation by Importance Weighted Cross Validation**, JMLR 8, 2007, 985-1005. [JMLR paper](https://www.jmlr.org/papers/v8/sugiyama07a.html).
   - Used for: the principle that ordinary validation can become biased when the state distribution shifts. Phase 4G diagnoses shift but does not yet importance-weight policy returns.

5. Isaac Gibbs and Emmanuel Candes, **Adaptive Conformal Inference Under Distribution Shift**, NeurIPS 2021. [Proceedings paper](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html).
   - Used for: treating interval coverage as a dynamic transfer diagnostic. Phase 4G measures frozen interval coverage; adaptive conformal updating remains a later hypothesis.

6. David H. Bailey and Marcos Lopez de Prado, **The Deflated Sharpe Ratio**, Journal of Portfolio Management 40(5), 2014. [SSRN paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551).
   - Used for: multiple-testing-aware policy promotion.

7. David H. Bailey, Jonathan M. Borwein, Marcos Lopez de Prado, and Qiji Jim Zhu, **The Probability of Backtest Overfitting**, Journal of Computational Finance 20(4), 2017. [SSRN paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253).
   - Used for: PBO across the four predeclared coverage policies.

## Phase 4G Boundary

- Transformer weights, embeddings, neighbours, outcome semantics, and A2 opportunity scores remain frozen.
- Reliability is fitted on 2021-2022 training states and tested on 2023 states.
- The chronological calibration tail is separated from model fitting by a 63-session embargo.
- Coverage thresholds come from the training calibration tail, never from validation quantiles.
- Test and holdout remain sealed unless every diagnostic and trading gate passes.
