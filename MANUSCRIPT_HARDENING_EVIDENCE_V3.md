# Manuscript hardening from compact VM evidence v3

## Evidence boundary

This revision uses only the compact evidence bundle at:

`C:\Users\rohil\Downloads\core_rl_paper_evidence (3)\rl`

The following original files are available in that bundle:

- `original_summaries/experiment_manifest.yaml`
- `original_summaries/build_summary.json`
- `original_summaries/market_audit_summary.csv`
- `original_summaries/leaderboard.csv`
- `original_summaries/market_results.csv`
- `original_summaries/selection.json`
- `original_summaries/confirmation_report.md`

The representation-audit artifacts, principal Git commit, detailed external-evaluation tables, exclusion reasons, data-quality ledger, currency-pooling record, resolved generated configurations, and observed software/hardware environment are not present. Claims requiring those artifacts are softened or declared unavailable below.

## Mandatory scientific correction

The original testbed did not promote or lock a policy candidate. The build contained 551 jobs: 7 data, 61 pilot, 483 full, and 0 confirmation jobs. Confirmation was disabled. The selection record states:

- status: rejected;
- selected candidate: none;
- best observed candidate: `global_global__coverage_025`;
- candidate count: 12;
- development split only: true;
- Transformer retrained: false;
- RL used: false.

The later 2025-2026 result must therefore be described as an **external temporal robustness evaluation of the best-observed rejected development candidate**, not as the sealed confirmation of a successfully promoted or locked model.

## Replacement abstract

Long-horizon equity decisions are difficult because market relationships change through time, outcomes are delayed, and a single return forecast does not expose the historical evidence or path risk supporting an entry. We study a transformer-memory framework in which a Patch Transformer encodes 252-session observations and an external causal memory retrieves only historical analogues whose outcomes are mature at the query date. A deterministic long-only policy ranks opportunities from weighted evidence describing upside, downside, path quality, agreement, diversity, and uncertainty. In a frozen 2024 international development sweep across 12 topology-and-coverage candidates, the best-observed global-encoder/global-memory configuration at 25% nominal reliability coverage achieved pooled Sharpe 1.503. Five of six markets were profitable, but China contributed 70.29% of positive profit. Importantly, no candidate passed the predeclared development promotion gates, so no principal candidate was promoted and no confirmation jobs were created by the original testbed. A later external temporal robustness evaluation of the best-observed candidate over 1 January 2025 to 31 March 2026 produced return -6.02%, Sharpe -0.376, maximum drawdown -13.59%, and only two profitable markets. The tested offline-reinforcement-learning path was not used in principal selection. The study therefore contributes an auditable causal-retrieval framework and a falsifiable evaluation showing that strong development-period results can coexist with failed promotion and failed temporal robustness. It does not establish a transferable profitable trading strategy.

Remove the numerical ARI and neighbour-association claims from the abstract until their original audit outputs are recovered.

## Replacement for Section 3.1 universe paragraph

The configured universe contained 18 requested equities in each of the United States, India, China, Brazil, France, and the United Kingdom. The market audit retained 18 securities in the United States, India, and China, 15 in Brazil, and 17 each in France and the United Kingdom, for 103 available securities from 108 requested. The compact evidence does not preserve the names or exact exclusion reasons for the five unavailable securities. The lists were manually fixed around liquid companies and were not reconstructed from historical point-in-time index membership; the study therefore remains subject to survivorship and researcher-selection bias.

### Retained-universe table

| Market | Requested | Available | Unavailable |
|---|---:|---:|---:|
| United States | 18 | 18 | 0 |
| India | 18 | 18 | 0 |
| China | 18 | 18 | 0 |
| Brazil | 18 | 15 | 3 |
| France | 18 | 17 | 1 |
| United Kingdom | 18 | 17 | 1 |
| **Total** | **108** | **103** | **5** |

Do not state the identities or causes of exclusion until a symbol-level ledger is recovered.

## Replacement for Section 3.7 representation audit paragraph

Representation reproducibility was investigated during project development, but the compact VM evidence currently available for manuscript restoration does not contain the original clustering assignments, pairwise adjusted Rand indices, association estimator, audit sample size, missing-data rules, or dependence-aware uncertainty estimates. Consequently, previously reported aggregate representation point estimates are treated as unverified historical summaries and are not used as confirmatory evidence in this version. The representation contribution is evaluated operationally through the downstream causal-retrieval experiment, while restoration of the original latent-audit artifacts is left as a reproducibility requirement.

Delete or replace Table 3. Do not report ARI approximately 0.9967 or associations 0.957, 0.969, and 0.877 as verified results from evidence pack v3.

## Replacement for Section 3.8 reproducibility controls

The principal testbed is identified as `phase6_a30_final_v1`. Its archived experiment manifest declares 551 jobs: 7 data jobs, 61 pilot jobs, 483 full jobs, and no confirmation jobs. The run required CUDA, one visible GPU, at least 6 GB VRAM, automatic mixed precision, and no CPU fallback. These values describe configured execution requirements rather than an observed hardware inventory. The compact manifest does not record the principal source commit, dirty-worktree status, run timestamps, actual GPU model, Python package lock, PyTorch/CUDA/cuDNN versions, or complete artifact hashes. These items must therefore be declared unavailable from the restored compact evidence rather than inferred from the current local repository or configuration templates.

## Replacement Table 4: Frozen 2024 development sweep

| Candidate | Pooled Sharpe | Positive markets | Worst MDD | Positive-profit concentration | Median trades | DSR probability | Passed gates |
|---|---:|---:|---:|---:|---:|---:|---|
| Global/global, 25% nominal coverage | 1.503026 | 5/6 | -21.71% | 70.29% | 59.0 | 0.4444 | No |
| Global/global, 75% nominal coverage | 1.277012 | 5/6 | -23.13% | 82.24% | 59.5 | 0.3553 | No |
| Global/global, 50% nominal coverage | 1.094225 | 5/6 | -31.68% | 78.95% | 56.0 | 0.2882 | No |
| Global/global, 100% nominal coverage | 0.532629 | 3/6 | -30.25% | 88.76% | 63.0 | 0.1298 | No |

All 12 candidates in the source leaderboard have `passes = false`. The 25% configuration is the best-observed candidate, not a promoted candidate.

## Replacement Table 5: Six-market results for the best-observed 2024 candidate

| Market | Return | Sharpe | Sortino | MDD | Calmar | Exposure | Trades | Win rate | Profit factor | Avg. hold | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| United States | 22.61% | 1.084 | 1.153 | -17.16% | 1.323 | 89.75% | 70 | 70.00% | 1.425 | 9.79 | 26.19 |
| India | 2.82% | 0.265 | 0.345 | -11.79% | 0.246 | 94.63% | 79 | 58.23% | 1.071 | 8.89 | 27.18 |
| China | 85.22% | 2.216 | 5.337 | -11.11% | 8.144 | 96.40% | 50 | 54.00% | 3.804 | 14.24 | 21.20 |
| Brazil | 4.41% | 0.315 | 0.534 | -11.41% | 0.390 | 97.99% | 45 | 53.33% | 1.120 | 16.60 | 15.27 |
| France | -16.53% | -0.901 | -1.434 | -21.71% | -0.753 | 89.14% | 56 | 44.64% | 0.628 | 12.36 | 17.43 |
| United Kingdom | 6.18% | 0.426 | 0.839 | -12.20% | 0.504 | 97.16% | 62 | 45.16% | 1.174 | 12.00 | 22.84 |

Turnover is reported exactly in the units emitted by the original source table. Do not append a percent sign until the implementation contract is confirmed.

## Correct reliability results

The derived `reliability_summary.csv` in the compact bundle is incorrectly mapped and must not be used. Use `original_summaries/market_results.csv` directly.

| Market | Realized coverage | Threshold | Brier | Brier skill | ECE | ROC AUC | Interval coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| United States | 0.2884 | 0.5075 | 0.2576 | -0.0319 | 0.0959 | 0.5181 | 0.5567 |
| India | 0.8701 | 0.4752 | 0.2573 | -0.0351 | 0.1705 | 0.4725 | 0.5201 |
| China | 0.9844 | 0.4619 | 0.2491 | -0.0007 | 0.0060 | 0.4976 | 0.5599 |
| Brazil | 0.9660 | 0.4577 | 0.2449 | 0.0124 | 0.0111 | 0.5206 | 0.4818 |
| France | 0.2348 | 0.4654 | 0.2568 | -0.0357 | 0.1156 | 0.4769 | 0.4625 |
| United Kingdom | 0.7030 | 0.4673 | 0.2446 | 0.0097 | 0.2622 | 0.5622 | 0.5667 |

Only Brazil and the United Kingdom had positive Brier skill, matching the leaderboard count of two. The nominal 25% label did not translate into approximately 25% realized coverage in four markets, which is an important reliability limitation. The file named `reliability_deciles.csv` is not a genuine decile table and must be excluded.

## Replacement for Section 4.2

The frozen 2024 development sweep evaluated 12 combinations of encoder topology, memory topology, and nominal reliability coverage. No candidate satisfied the complete promotion contract. The best-observed configuration was global encoder/global memory at 25% nominal coverage, with pooled Sharpe 1.503026, five profitable markets, worst market-level drawdown -21.71%, and positive-profit concentration of 70.29%. The nearby 75% configuration achieved pooled Sharpe 1.277012 and five profitable markets, but positive-profit concentration increased to 82.24%. The 25% candidate's deflated-Sharpe probability was 0.4444, and its robustness gate was false. These results establish a strong development-period observation but not successful promotion.

## Replacement for Section 4.3

Because no development candidate passed the promotion gates, the original testbed created no confirmation jobs and locked no principal candidate. A later external temporal robustness evaluation nevertheless assessed the best-observed `global_global__coverage_025` configuration over 1 January 2025 to 31 March 2026. It returned -6.02%, with pooled Sharpe -0.376, maximum drawdown -13.59%, and two profitable markets. China accounted for 89.60% of positive profit. The evaluation status was `external_evaluation_rejected`. The compact evidence does not include the underlying per-market external-evaluation table or sufficient source outputs to verify the manuscript's previously reported raw-memory, equal-weight, momentum, and direct-adapter comparison values. Those comparisons must be removed from Table 6 or explicitly labelled as unrestored until their original outputs are recovered.

### Replacement Table 6 using only restored evidence

| Evaluation | Candidate status | Return | Sharpe | MDD | Positive markets | Profit concentration |
|---|---|---:|---:|---:|---:|---:|
| External temporal robustness, 2025-01-01 to 2026-03-31 | Rejected | -6.02% | -0.376 | -13.59% | 2/6 | 89.60% |

Do not include raw memory, equal-weight, Momentum-21, or direct-adapter rows unless their original external-evaluation files are recovered.

## Replacement for offline-RL claims

The restored selection record states `rl_used = false`. It supports only the conclusion that offline RL was not part of the principal frozen-memory selection. It does not provide algorithm-level IQL or TD3+BC metrics. Replace claims that those algorithms were quantitatively rejected in this principal experiment with:

> Offline-RL branches were explored elsewhere in the research programme, but the restored principal selection record shows that RL was not used in the 12-candidate frozen-memory selection. Algorithm-level transfer claims are outside the evidence restored for this manuscript version.

## Replacement limitations additions

Add the following limitations:

1. The compact audit identifies 103 available securities from 108 requested, but it does not preserve symbol-level exclusion reasons, missing-price counts, eligible-window counts, or fundamental-missingness totals.
2. The original representation-audit outputs were not restored. Pairwise ARIs, association definitions, sample sizes, and uncertainty estimates are therefore unavailable and previously reported aggregate point estimates are not treated as verified confirmatory evidence.
3. The best-observed 2024 candidate failed every complete promotion decision recorded in the source leaderboard. The later 2025-2026 result is external robustness evidence for a rejected development candidate, not confirmation of a promoted model.
4. The nominal reliability coverage parameter did not produce consistent realized coverage across markets. Realized coverage ranged from 23.48% in France to 98.44% in China for the nominal 25% candidate.
5. The compact evidence does not preserve the principal source commit, observed execution environment, complete checkpoint/data hashes, currency-conversion contract, market-date pooling rule, or detailed external-evaluation comparison outputs.

## Replacement conclusion

We presented an auditable transformer-memory framework for long-horizon equity selection. The external memory enforces outcome maturity and exposes the historical precedents used by a deterministic long-only policy. In the frozen 2024 development sweep, the best-observed global-encoder/global-memory candidate achieved pooled Sharpe 1.503 and positive returns in five of six markets, but its profits were geographically concentrated and it failed the predeclared promotion gates. Accordingly, the original testbed selected no candidate and created no confirmation jobs. A later external temporal robustness evaluation of the best-observed rejected candidate returned -6.02%, Sharpe -0.376, and only two profitable markets. The restored evidence therefore supports the framework as an auditable research system and supports the methodological importance of explicit promotion and temporal-robustness gates. It does not support a claim of transferable profitability, successful reliability filtering, or a validated offline-RL trading policy.

## Replacement data and code availability statement

The research code and tracked configurations remain available in the project repository. The compact manuscript evidence preserves the principal experiment manifest, build summary, market audit, 12-candidate leaderboard, six-market development results, selection record, and aggregate external-evaluation report. Large raw data, full latent exports, checkpoints, query ledgers, and detailed trade ledgers are not distributed with the compact evidence. The compact record identifies their VM locations only partially and does not provide a complete verified hash inventory. Raw observations originated from Yahoo Finance and remain subject to provider terms and possible revision. Public or controlled-access status for each artifact category must be declared by the author before submission.

## Items that must remain unavailable in this revision

- Pairwise ARIs and dependence-aware ARI interval.
- Exact neighbour-association estimator, sample sizes, and intervals.
- Symbol-level exclusion reasons.
- Price-missingness and eligible-window accounting.
- Query rejection accounting.
- Detailed external-evaluation baselines and per-market results.
- Principal Git commit and dirty-worktree state.
- Observed VM software and hardware.
- Complete checkpoint and data hashes.
- Currency conversion and pooled-calendar treatment.

These must not be estimated from the PDF, current repository state, configured ticker lists, or rounded prose.
