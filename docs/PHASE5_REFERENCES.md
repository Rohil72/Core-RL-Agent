# Phase 5 Research References

These are the methods implemented or explicitly evaluated in Phase 5. Keeping
the list beside the code makes later interpretation independent of chat history.

## Representation alignment

- Yu et al., [Gradient Surgery for Multi-Task Learning (PCGrad)](https://arxiv.org/abs/2001.06782), NeurIPS 2020. Used to prevent the future-prediction and decision-neighborhood objectives from destructively interfering during the one conditional transformer pass.
- Bardes, Ponce, and LeCun, [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906), ICLR 2022. Used only as a low-weight anti-collapse regularizer for the 32-dimensional decision space.
- Nie et al., [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730), ICLR 2023. Relevant to the existing patch-token temporal encoder; Phase 5 does not enlarge that encoder.

## Offline policies

- Kumar et al., [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779), NeurIPS 2020. One of the three nonlinear offline-RL candidates.
- Kostrikov, Nair, and Levine, [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169), ICLR 2022. Evaluated on the same fixed observation/action/reward tuples as every other policy.
- Fujimoto and Gu, [A Minimalist Approach to Offline Reinforcement Learning (TD3+BC)](https://arxiv.org/abs/2106.06860), NeurIPS 2021. The behavior-regularized continuous-control candidate.
- Seno and Imai, [d3rlpy: An Offline Deep Reinforcement Learning Library](https://www.jmlr.org/papers/v23/22-0017.html), JMLR 2022. Runtime used for CQL, IQL, and TD3+BC in a separate environment.

## Backtest selection controls

- Bailey et al., [The Probability of Backtest Overfitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253), Journal of Computational Finance. Basis for the configured PBO rejection gate.
- Bailey and Lopez de Prado, [The Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551), Journal of Portfolio Management. Basis for the multiple-testing-aware Sharpe probability gate.
