# Prism AI Handoff: Semester VI Research Report Continuation

## Purpose and Scope

Use this document with the existing `Sem6 research report rough draft.pdf` as
the factual source for the next LaTeX pass. This is a continuation of the
earlier report, not a replacement for it. Preserve the front matter, the
long-horizon investment motivation, the formal academic tone, and the existing
literature-review table style.

For this pass, write only:

1. An updated Introduction.
2. An expanded Literature Survey.
3. A Results and Findings chapter.

Do not try to write the detailed methodology, architecture figure captions,
dataset tables, or implementation chapter yet. Those need diagrams and a
separate visual pass.

## Non-Negotiable Factual Corrections

The draft currently describes a final system based on reinforcement learning,
stage detection, news sentiment, and intent. That was the starting direction,
not the final implemented research outcome.

- News sentiment and intent were not part of the completed final experiments.
  Do not claim that they were integrated, evaluated, or improved performance.
  They may appear only as future work.
- Offline RL was evaluated late in the project using CQL, IQL, and TD3+BC. The
  tested formulation was rejected and is not the final decision policy.
- The Stage 1-4 / oracle cycle detector became a legacy weak-supervision and
  early-experiment tool. It is not the final market representation.
- The final research architecture is a patch Transformer market-state encoder,
  a causal searchable market memory, distributional evidence aggregation, and
  a deterministic long-only ranking policy.
- No trading policy is promoted for deployment. The report must not claim a
  robust profitable system, a market-agnostic 2+ Sharpe strategy, or a Q1-level
  empirical trading result.

Recommended revised title:

> Retrieval-Grounded Market Memory for Causal Long-Horizon Equity Decision
> Support

An acceptable continuity-preserving alternative is:

> From Stage-Aware Trading to Retrieval-Grounded Market Memory: A Causal
> Long-Horizon Equity Decision Framework

The second title makes the evolution from the previous report explicit.

## The Report's Correct Narrative Arc

The report should tell this sequence:

1. The project started from the sensible question of long-horizon, interpretable
   equity decisions. Early work used stage/cycle concepts and considered RL.
2. Latent-space audits showed that a patch Transformer learned reproducible,
   future-outcome-separated market states. This was stronger evidence than the
   manually specified stage detector.
3. The research therefore shifted from "predict a stage" to "retrieve similar
   historical states and reason over their realized outcomes."
4. A market-memory layer was built to retrieve causal historical analogues and
   estimate upside, downside, tail risk, holding period, agreement, and
   uncertainty.
5. The project then tested whether representation geometry, outcome semantics,
   exposure control, confidence filtering, rally prototypes, decision adapters,
   global/regional transfer, and offline RL made this evidence transferable.
6. Some development experiments were promising, including a static
   median-consensus C0 policy with out-of-fold Sharpe around 2.15. However,
   later causal audits, cross-market gates, and sealed confirmation showed that
   this was not sufficient evidence of a robust international trading strategy.
7. The main contribution is therefore a reproducible retrieval-grounded market
   memory architecture and a careful empirical record of what helped, what
   reduced risk only, and what failed under transfer.

Use cautious, research-appropriate verbs: "indicates", "supports",
"suggests", "was rejected by the declared gate", and "did not establish".
Avoid "proves", "guarantees", "consistently outperforms", and "profitable
agent".

## Introduction Material

### 1.1 Motivation

Keep the draft's motivation around non-stationarity, delayed outcomes,
drawdown, entry timing, capital efficiency, and the weakness of one-step price
forecasting. Update its central claim:

> Long-horizon equity decisions are not only a prediction problem. They require
> assessing whether the current market state resembles earlier situations that
> led to favorable or unfavorable outcome distributions, while respecting that
> market relationships and risk conditions shift through time.

Explain that a direct neural prediction head can compress historical experience
into a single score, whereas an analogue-memory design keeps the evidence
available for inspection. The desired question at each date is:

1. What market state is observed now?
2. Which historical states were genuinely similar and already outcome-matured?
3. What distribution of upside, downside, and path quality followed those
   precedents?

### 1.2 Relevance

Retain the long-horizon and low-turnover framing from the draft, but replace the
claim that RL is inherently the final solution. The relevance is instead:

- transparent evidence for entries rather than an opaque action score;
- realistic evaluation of downside, transaction costs, drawdown, and
  concentration;
- comparison across heterogeneous markets rather than a single tailored ticker
  universe;
- a modular system in which representation, memory, and policy can be audited
  separately.

### 1.3 Research Gap

Replace the draft's news-sentiment gap with these gaps:

1. Financial sequence models often optimize forecast loss without showing that
   their latent geometry retrieves economically comparable historical states.
2. A nearest-neighbor average discards disagreement, tail risk, density,
   holding-period, and path-quality information contained in retrieved cases.
3. Development backtests can look attractive while failing after temporal,
   geographic, or protocol transfer. Evaluation must therefore include causal
   outcome maturity, multiple-testing controls, per-market drawdown, and
   concentration diagnostics.
4. Learned confidence, abstention, and policy layers are often assumed to help;
   they need to be tested as falsifiable mechanisms rather than treated as
   automatic improvements.

### 1.4 Revised Problem Statement

> To develop and evaluate a causal, retrieval-grounded market-memory framework
> for long-horizon equity decision support. The framework encodes historical
> technical and point-in-time fundamental observations into market states,
> retrieves outcome-matured historical analogues, aggregates their realized
> outcome distributions, and converts evidence into interpretable long-only
> ranking decisions under realistic trading constraints.

### 1.5 Revised Objectives

Use three objectives, not the current news/RL objectives:

| Objective | Report wording |
|---|---|
| O1: Representation | Learn and audit a temporal representation that produces stable market states with measurable future-outcome separation. |
| O2: Historical reasoning | Build causal market memory that retrieves analogous prior states and estimates expected upside, downside, tail risk, holding period, and evidence agreement. |
| O3: Robust evaluation | Test whether memory-based decisions survive walk-forward, cross-market, and confirmation-style evaluation against simple deterministic baselines and declared robustness gates. |

### 1.6 Revised Hypotheses

Do not retain hypotheses that require news sentiment or an RL final policy.

| ID | Hypothesis | Evaluation interpretation |
|---|---|---|
| H1 | The encoder learns reproducible market-state structure associated with future opportunity profiles. | Cluster reproducibility, latent-neighbour outcome correlation, and ticker-diversity audit. |
| H2 | Causal retrieval evidence can be more useful than a direct prediction head for opportunity ranking. | Retrieval-versus-head backtest comparison and retrieval-quality diagnostics. |
| H3 | Outcome-aware evidence and multi-scale disagreement can improve risk awareness relative to simple neighbour averaging. | Upside/downside tails, stop-loss counts, drawdown, and rank-spread analysis. |
| H4 | A mechanism is considered robust only when it survives per-market and temporally separated gates, not merely pooled development Sharpe. | Market wins, drawdown, concentration, PBO/DSR, and confirmation results. |

H1 is supported with an important ticker-entanglement limitation. H2 is only
partially supported. H3 is partially supported as a risk mechanism, not as a
proven alpha engine. H4 is a protocol conclusion: several promising mechanisms
failed it.

## Literature Survey Material

Keep the current table-based literature survey, but add the following six
subsections after the existing discussion of market regimes and representation
learning. The literature review should lead naturally from fixed regimes toward
retrieval-grounded reasoning.

### 2.1 Market Regimes and State Discovery

Retain the existing Hamilton/regime-clustering discussion. Then explain the
project's position: market states are treated as empirical latent opportunity
profiles, not as a requirement to reproduce a fixed Stage 1-4 taxonomy. The
early detector provided weak supervision and research orientation, but later
latent-state results motivated a more flexible state representation.

### 2.2 Transformer Representation Learning for Financial Time Series

Connect the active patch-based encoder to PatchTST and the wider multivariate
time-series representation literature. Patch-based modeling is relevant because
it captures local temporal context while still allowing longer contextual
reasoning. Mention self-supervised reconstruction as an auxiliary stability
objective, not as a claim of a financial foundation model.

Use: `nie2023patchtst`, `zerveas2021transformer`, `yue2022ts2vec`,
`li2023timae`, and `lim2021tft`.

### 2.3 Retrieval-Augmented Forecasting and Historical Market Memory

Introduce the key departure from a feed-forward prediction head: historical
states are stored with later realized outcomes, then retrieved only once their
outcomes are mature. Discuss retrieval augmentation as a way to ground a
decision in precedents, while making clear that the system is not an LLM and
does not claim formal conformal guarantees.

Use: `borgeaud2022retro`, `yang2022mqretcnn`, `raft2025`, `raf2024`, and
`xiao2025financialrag`.

### 2.4 Sequential Uncertainty, Calibration, and Selective Decisions

Explain why the memory layer returns a distribution, not only a mean. The
project uses empirical tails, neighbour agreement, effective sample size,
entropy, diversity, and distance information as evidence diagnostics. These are
inspired by sequential uncertainty and calibration research, but they are not
presented as calibrated probabilities unless proven as such.

Use: `xu2021conformal`, `zaffran2022adaptive`, `xu2023sequential`,
`guo2017calibration`, `geifman2017selective`, `gibbs2021adaptive`,
`rabanser2019shift`, and `sugiyama2007covariate`.

### 2.5 Long-Horizon Entry, Momentum, and Rally Outcomes

Frame the market-memory targets as outcome-aware rather than as an attempt to
replicate a named technical-analysis rule. The literature motivates testing
medium-horizon continuation and risk, while the project operationalizes this
through maximum favorable excursion (MFE), maximum adverse excursion (MAE),
relative alpha, and holding-path measures.

Use: `jegadeesh1993momentum`, `moskowitz2012tsmom`,
`he2020earningsacceleration`, `lara2021`, and `sthan2021`.

### 2.6 Robust Backtesting, Risk, and Offline RL Baselines

Explain that risk overlays and offline RL were evaluated as branches, not
assumed final components. PBO and DSR are used to temper selection claims.
Exposure controls were informed by volatility management and drawdown work but
did not create reliable cross-market alpha. CQL, IQL, and TD3+BC are reported
as evaluated baselines whose tested formulation was rejected.

Use: `bailey2014dsr`, `bailey2017pbo`, `moreira2017volatility`,
`barroso2015momentum`, `daniel2016crashes`, `chekhlov2005drawdown`,
`kumar2020cql`, `kostrikov2022iql`, and `fujimoto2021td3bc`.

### 2.7 Literature Synthesis and Positioning

End the survey with this positioning:

> The proposed contribution is not a claim that a new deep policy universally
> beats markets. It is a causal, modular market-memory framework that connects
> learned temporal states with inspectable historical analogues and evaluates
> whether the resulting evidence survives temporal and geographic transfer.

This sentence is the bridge from the previous report to the completed work.

## Results and Findings Chapter Material

Suggested chapter title:

> Chapter 4: Experimental Findings and Research Evolution

Open the chapter by explaining that this is an ablation-led research record.
Some early performance figures are retained to explain hypothesis evolution, but
are explicitly marked as development-only or potentially contaminated where a
later protocol audit found a flaw. The report should treat rejected results as
findings, not hide them.

### 3.1 Result Status Legend

Use these labels in tables:

- Established: supported by recorded result and artifact.
- Partial: useful evidence exists, but a major limitation remains.
- Rejected: failed its predeclared promotion or robustness gate.
- Diagnostic-only: already inspected period; not fresh confirmation evidence.
- Potentially contaminated: later timestamp-precision audit affects causal
  interpretation until regenerated.

### 3.2 Representation Discovery

This is the strongest positive result and should be presented first.

| Metric | Recorded result | Interpretation |
|---|---:|---|
| KMeans cluster reproducibility, k=8 | Adjusted Rand Index about 0.9967 across seeds | Latent organization was highly reproducible. |
| Neighbour correlation with future maximum return | about 0.957 | Nearby states shared upside profiles. |
| Neighbour correlation with future minimum return | about 0.969 | Nearby states shared downside profiles. |
| Neighbour correlation with path quality | about 0.877 | The latent captured path-related outcome structure. |
| Ticker entropy | below random reference | State geometry was not fully ticker-invariant. |

Conclusion: the representation hypothesis is supported, but the latent space is
not a universal company-agnostic market-state map.

### 3.3 Early Retrieval Proof and Retrieval-Semantics Ablation

| Experiment | Result | Correct conclusion |
|---|---|---|
| Early memory harness | Retrieval policy: +23.8% return, Sharpe 0.889, profit factor 1.63; direct model head: -17.4% return | Retrieval contained useful development-period information beyond the direct head. It was not cross-market confirmation. |
| Phase 4D A0 identity | mean return 20.1%, mean Sharpe 0.950 | Reference retrieval semantics. |
| Phase 4D A1 universe alpha | mean return 29.2%, mean Sharpe 1.430 | Relative outcomes improved policy semantics. |
| Phase 4D A2 blended alpha | mean return 36.7%, mean Sharpe 1.607 | Best return in the ladder. |
| Phase 4D A3 diagonal metric | mean return 33.8%, mean Sharpe 1.661 | Best Sharpe but did not improve retrieval gain. |
| Phase 4D A4 low-rank metric | mean return 24.6%, mean Sharpe 1.155 | Seed-sensitive. |

Both A2 and A3 failed their paired-bootstrap promotion condition. Mark this
whole Phase 4D table as potentially contaminated because it predates the later
nanosecond timestamp repair.

### 3.4 Memory, Risk, and Rally-Entry Ladder

| Phase | Main finding | Status |
|---|---|---|
| 4E cross-market memory | Best candidate Sharpe 1.920; worst drawdown -32.6%; no promotion. | Partial, potentially contaminated. |
| 4F exposure control | Best in-sample Sharpe 1.961; worst drawdown improved to about -24.9%; return and robust alpha did not improve. | Risk improvement, no promotion. |
| 4G reliability | Selected 75% coverage Sharpe 1.331; OOF Sharpe 0.853; Brier skill negative in all folds. | Rejected. |
| 4H rally prototypes | Sharpe fell from 1.921 to 1.640 while drawdown improved from -32.6% to -24.1%; rally AUC about 0.503. | Rejected as an entry-alpha signal. |
| 4I neutral memory | Rally-entry precision 53.4% to 56.1%; worst drawdown -32.6% to -19.5%; Sharpe 1.921 to 1.797. | Useful abstention/risk diagnostic, not alpha engine. |

Key lesson: several mechanisms reduced risk, but none established transferable
alpha. This is a valuable result, because it prevents the report from confusing
lower drawdown with a successful predictive mechanism.

### 3.5 Decision Adapter and Opportunity Allocation

| Result | Recorded value | Interpretation |
|---|---:|---|
| Adapter promotion variants | OOF Sharpe 1.651 and 1.722 | Promising but failed full contract; worst drawdown about -32.8% in the stronger variant. |
| Static C0 median consensus | OOF Sharpe about 2.15; PBO about 0.07 | Strongest development-only result; not attributable to later exposure control. |
| Corrected static allocator B0 | Sharpe 2.123 | Existing deterministic signal remained strong in development. |
| Deterministic obvious signal B1 | Sharpe 2.149; drawdown about 5.1 percentage points lower than B0 | Improved development risk/return tradeoff, but failed out-of-year ranking. |
| Final allocator | zero out-of-year positive folds; PBO about 0.543 | Rejected; calibration and nonlinear allocation did not transfer. |

The timestamp-precision audit is important here: corrected causal evidence had
five folds, 28 seed runs, and zero causal violations. It repaired the evidence
construction but did not rescue the allocator's out-of-year performance.

### 3.6 International Evaluation and Offline RL

| Candidate | Market wins | Median Sharpe | Worst drawdown |
|---|---:|---:|---:|
| IQL, global | 4 | 0.057 | -0.345 |
| IQL, regional | 3 | -0.077 | -0.292 |
| TD3+BC, global | 3 | 0.296 | -0.241 |
| TD3+BC, regional | 1 | 0.291 | -0.378 |

The Phase 6 selection gate rejected every RL candidate, with reported PBO 0.714.
Explain the scope carefully: this rejects the tested per-seed, one-day-reward
offline allocation formulation. It does not logically reject every possible use
of RL, but it was removed from the active system because it added complexity
without stable improvement.

The no-RL frozen memory topology sweep is the correct place to report the
approximately 1.2 all-market aggregate the project reached. It produced two
important development-only reference points:

| Global encoder + global memory configuration | Pooled Sharpe | Median market Sharpe | Positive markets | Worst drawdown | Profit concentration |
|---|---:|---:|---:|---:|---:|
| 25% reliability coverage | 1.503 | 0.370 | 5/6 | -21.71% | 70.3% in China |
| 75% reliability coverage | 1.277 | 0.326 | 5/6 | -23.13% | 82.2% in China |

The 75% configuration is the appropriate source for the remembered aggregate
Sharpe of about 1.2. It was not an equally strong result in every market: it
won against the declared baseline in only 3/6 markets, its median market Sharpe
was low, and its profits were heavily concentrated in China. Both configurations
were therefore rejected by the robustness contract.

The sealed confirmation is the most important negative result:

| Strategy | Return | Sharpe | Positive markets |
|---|---:|---:|---:|
| Locked reliability-filtered memory | -6.02% | -0.376 | 2/6 |
| Raw memory | +2.37% | 0.218 | 3/6 |
| Equal-weight buy and hold | +9.01% | 0.666 | 3/6 |

The locked reliability policy concentrated 89.6% of profit in China and raw
memory beat it in five of six markets. This is why the final report must not
present learned reliability filtering or RL as validated improvements.

### 3.7 Final Enriched Memory Study

The final no-retraining memory study explicitly compared raw 128-dimensional
encoder memory, 32-dimensional adapter memory, rally-path evidence, and
multi-scale growing memory.

| Variant | Development Sharpe | Selection Sharpe | Observed Sharpe |
|---|---:|---:|---:|
| Raw C0 | -0.157 | -0.037 | 0.322 |
| Adapter C0 | 0.387 | -0.586 | -0.471 |
| Rally static | 0.415 | 0.412 | 0.116 |
| Rally multiscale | 0.409 | 0.450 | 0.351 |

The multiscale rally variant was the most stable enriched construction and
reduced stop-loss trades from 40 to 14 in selection and from 65 to 28 in the
observed period relative to static rally memory. It also showed a positive
top-versus-bottom rank alpha spread across the three periods.

However, it did not establish robust international alpha: the best observed
pooled Sharpe was only 0.351, positive markets fell to 2/6, and worst-market
drawdowns were severe. The original study also had two validity defects:

1. An unbounded path-quality ratio could dominate return evidence.
2. International inference coverage was incomplete for some observed-period
   markets, especially India.

The closing repair protocol now bounds path quality to [0,1], enforces
nanosecond causal timestamps, audits inference coverage, and compares only raw
C0, bounded rally-static, and bounded rally-multiscale variants. It is still an
internal diagnostic. Do not report a result for it until it completes, and do
not call it external confirmation because the periods have already been
inspected.

### 3.8 Aggregation and Evidence Hierarchy

Do not end the Results chapter with the final enriched-memory table as though it
were the single final score. The project contains several pooled values under
different contracts, and they must not be treated as directly interchangeable.
Use this short synthesis table near the end of Chapter 4:

| Evidence context | Pooled Sharpe | What it establishes | Limitation |
|---|---:|---|---|
| Static C0 median-consensus development result | about 2.15 | Strongest corrected development-only C0 finding | Not external confirmation; later controllers did not create it. |
| Global/global memory, 25% coverage development sweep | 1.503 | Strongest pooled no-RL topology result | 70.3% China profit concentration and drawdown gate failure. |
| Global/global memory, 75% coverage development sweep | 1.277 | Broad six-market pooled reference closest to the report's aggregate endpoint | 82.2% China concentration, only 3/6 baseline wins, and low median market Sharpe. |
| Sealed reliability-filter confirmation | -0.376 | Reliability filter failed transfer | Not a positive final result. |
| Final enriched-memory observed diagnostic | 0.351 | Multi-scale memory reduced stop-loss entries | Path-quality and coverage defects; not a clean endpoint. |

Explain in one paragraph that a pooled Sharpe is calculated from the combined
portfolio return stream and is not the arithmetic average of six market Sharpes.
It can be dominated by a large or unusually profitable market. For that reason,
every pooled value in the report must be paired with median market Sharpe,
positive-market count, drawdown, baseline wins, and profit concentration.

### 3.9 Results Discussion and Honest Conclusion

Use this as the chapter conclusion:

> The experiments support the existence of reproducible, outcome-associated
> latent market states and show that historical retrieval can expose useful
> opportunity information. They do not yet support a claim of a robust,
> profitable, market-agnostic trading strategy. Risk overlays, reliability
> filters, and offline RL frequently reduced apparent development risk while
> failing to preserve transferable alpha. The strongest contribution is the
> causal market-memory architecture and the evidence-driven experimental record
> that separates stable representation findings from unconfirmed trading claims.

## Facts Prism Must Keep Separate

| Category | What may be claimed | What must not be claimed |
|---|---|---|
| Latent space | Stable and outcome-associated in recorded Phase 3 audits. | Fully ticker-invariant or universally transferable. |
| Retrieval | Useful development-period information; initially beat the direct head. | Consistent international profitability. |
| 2.15 Sharpe result | Best development-only C0 median-consensus finding. | Validated final performance or proof of the later controllers. |
| Risk layers | Sometimes reduced drawdown and stop-loss exposure. | Created reliable alpha. |
| RL | Evaluated as a baseline branch and rejected for the tested formulation. | Final policy or a generally failed research field. |
| 2025-2026 Q1 | Observed diagnostic evidence. | Fresh untouched confirmation. |
| Final repair | Frozen diagnostic protocol underway. | Completed result or deployment candidate. |

## Reference Pack for Prism

The existing rough draft already contains a 25-item literature list. Preserve
its regime, financial time-series, momentum, weak-supervision, and forecasting
references where they remain relevant. Add or ensure the following references
appear with the provided citation keys. Use one consistent IEEE or author-year
style throughout; do not duplicate a paper that already appears in the old
bibliography.

| Citation key | Reference and use in this report |
|---|---|
| `nie2023patchtst` | Nie, Nguyen, Sinthong, and Kalagnanam. *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.* ICLR, 2023. Patch-based temporal encoder. https://arxiv.org/abs/2211.14730 |
| `zerveas2021transformer` | Zerveas et al. *A Transformer-based Framework for Multivariate Time Series Representation Learning.* KDD, 2021. Representation-learning context. |
| `yue2022ts2vec` | Yue et al. *TS2Vec: Towards Universal Representation of Time Series.* AAAI, 2022. Temporal representation baseline context. |
| `li2023timae` | Li et al. *Ti-MAE: Self-Supervised Masked Time Series Autoencoders.* arXiv:2301.08871, 2023. Auxiliary reconstruction context. |
| `lim2021tft` | Lim et al. *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.* International Journal of Forecasting, 2021. Interpretable forecasting comparator context. |
| `borgeaud2022retro` | Borgeaud et al. *Improving Language Models by Retrieving from Trillions of Tokens.* ICML, 2022. General retrieval-augmented reasoning motivation. |
| `yang2022mqretcnn` | Yang, Eisenach, and Madeka. *MQ-ReTCNN: Multi-horizon Time Series Forecasting with Retrieval Augmentation.* 2022. Historical-analogue forecasting context. |
| `raft2025` | *Retrieval Augmented Time Series Forecasting.* arXiv:2505.04163, 2025. Retrieval-time-series motivation. https://arxiv.org/abs/2505.04163 |
| `raf2024` | *Retrieval-Augmented Forecasting.* arXiv:2411.08249, 2024. Retrieval-based forecasting motivation. https://arxiv.org/abs/2411.08249 |
| `xiao2025financialrag` | Xiao et al. *Enhancing Financial Time-Series Forecasting with Retrieval-Augmented Large Language Models.* arXiv:2502.05878, 2025. Financial retrieval context only; do not claim this project uses an LLM. |
| `xu2021conformal` | Xu and Xie. *Conformal Prediction for Time Series.* ICML, 2021. Sequential uncertainty context. https://proceedings.mlr.press/v139/xu21h.html |
| `zaffran2022adaptive` | Zaffran et al. *Adaptive Conformal Predictions for Time Series.* ICML, 2022. Non-stationary uncertainty context. https://proceedings.mlr.press/v162/zaffran22a.html |
| `xu2023sequential` | Xu and Xie. *Sequential Predictive Conformal Inference for Time Series.* ICML, 2023. Sequential coverage context. https://proceedings.mlr.press/v202/xu23r.html |
| `guo2017calibration` | Guo et al. *On Calibration of Modern Neural Networks.* ICML, 2017. Calibration diagnostics. https://proceedings.mlr.press/v70/guo17a.html |
| `geifman2017selective` | Geifman and El-Yaniv. *Selective Classification for Deep Neural Networks.* NeurIPS, 2017. Risk-coverage and abstention context. |
| `rabanser2019shift` | Rabanser, Guennemann, and Lipton. *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.* NeurIPS, 2019. Distribution-shift diagnostics. |
| `sugiyama2007covariate` | Sugiyama, Krauledat, and Mueller. *Covariate Shift Adaptation by Importance Weighted Cross Validation.* JMLR, 2007. Shift-aware validation motivation. |
| `gibbs2021adaptive` | Gibbs and Candes. *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS, 2021. Dynamic coverage context. |
| `jegadeesh1993momentum` | Jegadeesh and Titman. *Returns to Buying Winners and Selling Losers.* Journal of Finance, 1993. Momentum baseline context. |
| `moskowitz2012tsmom` | Moskowitz, Ooi, and Pedersen. *Time Series Momentum.* Journal of Financial Economics, 2012. Long-horizon momentum context. |
| `he2020earningsacceleration` | He and Narayanamoorthy. *Earnings Acceleration and Stock Returns.* Journal of Accounting and Economics, 2020. Point-in-time fundamental context. |
| `bailey2014dsr` | Bailey and Lopez de Prado. *The Deflated Sharpe Ratio.* Journal of Portfolio Management, 2014. Selection-adjusted Sharpe. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 |
| `bailey2017pbo` | Bailey, Borwein, Lopez de Prado, and Zhu. *The Probability of Backtest Overfitting.* Journal of Computational Finance, 2017. PBO gate. https://doi.org/10.21314/JCF.2016.322 |
| `moreira2017volatility` | Moreira and Muir. *Volatility-Managed Portfolios.* Journal of Finance, 2017. Exposure-control motivation. |
| `barroso2015momentum` | Barroso and Santa-Clara. *Momentum Has Its Moments.* Journal of Financial Economics, 2015. Time-varying risk motivation. |
| `daniel2016crashes` | Daniel and Moskowitz. *Momentum Crashes.* Journal of Financial Economics, 2016. Tail-risk motivation. |
| `chekhlov2005drawdown` | Chekhlov, Uryasev, and Zabarankin. *Drawdown Measure in Portfolio Optimization.* International Journal of Theoretical and Applied Finance, 2005. Drawdown evaluation. |
| `kumar2020cql` | Kumar et al. *Conservative Q-Learning for Offline Reinforcement Learning.* NeurIPS, 2020. Evaluated RL baseline. https://arxiv.org/abs/2006.04779 |
| `kostrikov2022iql` | Kostrikov, Nair, and Levine. *Offline Reinforcement Learning with Implicit Q-Learning.* ICLR, 2022. Evaluated RL baseline. https://arxiv.org/abs/2110.06169 |
| `fujimoto2021td3bc` | Fujimoto and Gu. *A Minimalist Approach to Offline Reinforcement Learning.* NeurIPS, 2021. TD3+BC baseline. https://arxiv.org/abs/2106.06860 |
| `bardes2022vicreg` | Bardes, Ponce, and LeCun. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.* ICLR, 2022. Low-weight anti-collapse objective in an experimental branch. |
| `yu2020pcgrad` | Yu et al. *Gradient Surgery for Multi-Task Learning.* NeurIPS, 2020. Experimental multi-objective alignment context. |

The rally-prototype references `LARA` and `STHAN-SR` may be cited in the
rally-entry subsection only if Prism resolves complete bibliographic metadata
from the URLs in `docs/research_record/REFERENCES.md`. Do not invent authors or
venues from shorthand titles.

## Repository Sources for Later Methodology and Diagrams

When the next report pass begins, use these sources rather than re-deriving the
system from memory:

- `ARCHITECTURE.md` for model dimensions, activations, loss configuration, and
  the final component graph.
- `docs/research_record/00_evidence_boundary.md` for causal and confirmation
  limits.
- `docs/research_record/03_phase3_latent_state_discovery.md` for stable latent
  state results.
- `docs/research_record/04_phase4a_c_encoder_and_holdouts.md` through
  `10_final_memory_study.md` for chronological experimental results.
- `docs/research_record/11_consolidated_findings.md` for the final scientific
  conclusion.
- `docs/research_record/12_q1_readiness.md` for publication-quality boundaries.
- `docs/FINAL_MEMORY_REPAIR.md` for the current closing diagnostic protocol.

## Final Instructions to Prism

1. Preserve the continuity of the original report: the early stage/RL framing
   is historical context, not an error to erase.
2. Update the present-tense system description to Transformer + causal memory +
   deterministic policy.
3. Include rejected and negative findings prominently. They make the work more
   credible, not less.
4. Put all numerical results in clearly labelled development, selection,
   observed-diagnostic, or sealed-confirmation categories.
5. Do not fabricate final-repair results, transaction-cost sensitivity results,
   news results, live-trading results, or factor-alpha results.
6. Keep citations precise. A cited paper motivates a design choice; it does not
   mean this project reproduced that paper's dataset or result.
