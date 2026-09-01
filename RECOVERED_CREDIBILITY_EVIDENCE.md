# Recovered Credibility Evidence Register

**Project:** Core-RL-Agent  
**Prepared:** 26 August 2026  
**Purpose:** Record the experiment-provenance, implementation, and result evidence that is now locally recoverable. This is an evidence-recovery dossier, not a claim that the research result has been validated.

## 1. Evidence basis and status language

This dossier was prepared from the following local materials:

- the complete reviewer bundle, `C:\Users\rohil\Downloads\core_rl_complete_paper_evidence\manuscript_evidence` (8,256 files; approximately 6.18 GB);
- the compact principal-run pack, `C:\Users\rohil\Downloads\core_rl_paper_evidence (3)\rl`;
- the tracked Git history and current source repository.

The status labels used below are deliberate:

| Status | Meaning |
|---|---|
| **Recovered** | The original artefact or an exact historical source version is locally available. |
| **Reconstructable** | The historical implementation is recoverable from an exact source commit and retained contract, but a primary run artefact is absent. |
| **Partial** | Important fragments exist, but they do not prove the whole claim. |
| **Unavailable** | The required primary evidence is not in the local materials. |
| **Not repairable by recovery** | This is an experimental/scientific limitation, not a documentation gap. |

## 2. Executive conclusion

The local materials recover substantially more than the former compact-pack review allowed. In particular, they recover the historical source baseline, the full job manifest, core run contracts, execution records, detailed trade/evaluation outputs, and multiple baseline/RL artefact families.

They do **not** recover principal checkpoint binaries, optimizer state, immutable raw data snapshots, provider retrieval timestamps, release-time records, the principal dirty-tree diff, all resolved generated configurations, or the original latent-representation audit outputs. A fresh prospective evaluation cannot be recovered from archival files.

Therefore, this evidence can support a far stronger reproducibility appendix and a precise methods/results reconstruction. It cannot convert the historical development result into a validated profitable strategy or turn an already inspected interval into prospective confirmation.

## 3. Principal-run provenance

### 3.1 Identified principal run

| Field | Recovered value | Status |
|---|---|---|
| Run ID | `phase6_a30_final_v1` | Recovered |
| Run root | `reports/final_testbed/phase6_a30_final_v1` | Recovered |
| Manifest SHA-256 | `a057cefde64269490221d6ddb514cf4a8034ccae5bb5249f4a6de0009679304d` | Recovered |
| Post-migration source fingerprint | `3237719dff40e94e9d31a42c40be2002d770cbbff2a10e907888684bff149c59` | Recovered |
| Historical source commit | `23922607a8d45c47c198fde609f0f046440231f7` | Recovered with a narrow qualification |
| Commit subject/date | `lossless continuation changes`; 24 July 2026, 23:00:47 +05:30 | Recovered |

The recovered source commit is supported by a file-level comparison against the run contract: **128 of 132** source files have an exact SHA-256 match. The four nonmatching files are helper/inspection utilities:

- `scripts/_count_valid.py`
- `scripts/_inspect_latent_sample.py`
- `scripts/_inspect_latents.py`
- `scripts/_make_test_small.py`

The source commit should therefore be reported as the **recovered principal source baseline**, while retaining the qualification that the four helper-script blobs and the original dirty-worktree diff are not proven.

### 3.2 Manifest and execution plan

The retained principal manifest is available at:

`C:\Users\rohil\Downloads\core_rl_paper_evidence (3)\rl\original_summaries\experiment_manifest.yaml`

It defines **551 jobs**:

| Stage | Jobs |
|---|---:|
| Data | 7 |
| Pilot | 61 |
| Full | 483 |
| Confirmation | 0 |

Confirmation was explicitly disabled. The manifest preserves each planned command, working inputs, dependencies, resource requirements, and expected output path. It is therefore sufficient to reconstruct the intended launch DAG, although it is not a terminal transcript proving every shell invocation exactly as executed.

### 3.3 Environment and training execution evidence

Thirty `training_complete.json` records are retained under the principal run. A representative principal record (`models/global_seed_7/training_complete.json`) reports:

| Field | Observed value |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.2.2+cu121 |
| CUDA runtime | 12.1 |
| cuDNN | 8902 |
| Device | NVIDIA A30 |
| Compute capability | 8.0 |
| VRAM | 25,337,004,032 bytes |
| AMP / CPU fallback policy | AMP enabled; CPU fallback disabled |
| Checkpoint interval | 250 steps |

These are observed runtime records for completed training jobs. They are stronger than template configuration, but they are not a complete principal-run `pip freeze` or container image.

### 3.4 AMP overflow recovery

The migration record `orchestration_state/migrations/0001_source_migration.json` states that isolated AMP gradient overflows were recovered by allowing `GradScaler` to skip the affected update and reduce scale. It records:

- prior source fingerprint: `e4ec5326716c38871dbcbecf93892034aca18828a1ed46ab3afd74046e011df0`;
- post-migration source fingerprint: `3237719dff40e94e9d31a42c40be2002d770cbbff2a10e907888684bff149c59`;
- the three altered files: `scripts/run_durable_experiment.py`, `src/orchestration/durable_runner.py`, and `src/trainers/train_cycle_model.py`;
- 82 completed jobs preserved and one job reset to pending.

This resolves the existence and mechanism of the overflow recovery. It does not by itself prove that no numerical output changed; that must be assessed from the affected-job/output mapping.

## 4. Recovered modelling and decision contract

The historical source baseline makes the following material reconstructable from implementation rather than prose:

| Contract area | Historical source of truth | Status |
|---|---|---|
| 28 feature formulas and transformations | `src/data/features.py` at commit `2392260` | Reconstructable |
| Data loading, alignment, missing-value handling, sequence construction | `src/data/loader.py`, `src/data/sequence_dataset.py` | Reconstructable |
| 11-target vector, masks, geometry and loss construction | `src/losses/outcome_geometry.py`, `src/decision/trainer.py` | Reconstructable |
| Retrieval eligibility, distance, weighting, trimming and aggregation | `src/memory/retrieval.py`, `src/memory/aggregator.py`, `src/memory/confidence.py` | Reconstructable |
| Reliability score and calibration diagnostics | `src/eval/memory_reliability.py`, `src/memory/confidence.py` | Reconstructable |
| Opportunity score, policy ranking and exits | `src/policy/opportunity_allocator.py`, `src/backtest/market_memory_backtester.py` | Reconstructable |
| Costs, trade accounting and market-level evaluation | `src/backtest/market_memory_backtester.py`, `src/backtest/market_memory_evaluator.py` | Reconstructable |
| Promotion statistics and selection logic | `src/eval/statistical_promotion.py`, `scripts/select_final_policy.py` | Reconstructable |

The exact principal training record identifies the 28 feature names and all 11 outcome targets. The target vector is:

`future_return_21`, `future_return_63`, `future_return_126`, `future_max_return_63`, `future_max_return_252`, `future_min_return_63`, `event_peak_offset_63`, `event_drawdown_offset_63`, `event_upside_before_drawdown_126`, `event_upside_hit_126`, and `event_drawdown_hit_126`.

### Important implementation qualification

The current checkout must not be cited as if it were the historical run source. Its source fingerprint differs from the principal run in 58 of the 132 contracted files. Method definitions must be extracted from `git show 23922607a8d45c47c198fde609f0f046440231f7:<path>` or from an archival checkout of that commit.

## 5. Promotion and model-selection contract

### 5.1 Recovered configured gates

The exact historical `configs/final_research_testbed.yaml` blob matches the run contract. It declares:

| Gate | Value |
|---|---:|
| Target pooled Sharpe | 2.0 |
| Maximum drawdown | 20% |
| Minimum positive markets | 5 |
| Maximum PBO | 0.50 |
| Maximum positive-profit concentration | 35% |
| Non-degenerate actions | Required |

The recovered `select_final_policy.py` computes PBO using the retained statistical-promotion implementation and applies a Boolean selection rule requiring, at minimum, sufficient market wins, nondegenerate markets, worst drawdown within the configured maximum, positive median excess Sharpe, and PBO at or below the configured maximum. Any manuscript should separately report the configured promotion gates and the actual selection-program Boolean logic rather than silently merging them.

### 5.2 Recovered selection outcome

The complete 12-candidate leaderboard and selection record are retained in the compact pack.

| Field | Recovered value |
|---|---|
| Selection status | `rejected` |
| Selected candidate | None |
| Best observed candidate | `global_global__coverage_025` |
| Best observed pooled Sharpe | 1.5030264573 |
| Candidate count | 12 |
| PBO | 0.0 |
| Development-only selection | True |
| Transformer retrained | False |
| RL used in selection | False |
| Confirmation jobs created | 0 |

Every candidate has `passes = false` in the retained leaderboard. This is a recovered result, not a missing-record issue.

## 6. Recovered evaluation artefacts

The complete reviewer bundle contains the following machine-readable evidence groups:

| Artefact group | Count |
|---|---:|
| Per-run metric files | 3,449 |
| Trade ledgers | 3,209 |
| Retrieval signal tables | 1,003 |
| Neighbour ledgers | 534 |
| Training completion records | 30 |
| Parquet tables overall | 1,537 |

The artefacts cover the principal final testbed, frozen-memory sweep, final-memory study/repair/confirmation families, transfer-credibility work, action-memory studies, and temporal-transport experiments. Directory names preserve run family, market, seed, topology/variant, and evaluation split; these should be retained in any output-to-job mapping.

### 6.1 Baselines and offline-RL artefacts

Later workflow families include raw-memory, locked-memory, direct-adapter, Momentum-21, equal-weight buy-and-hold, random-ranking, IQL, and TD3+BC files. These can support baseline/ablation tables **only when labelled with their actual run family and split**. They must not be retroactively presented as baselines for the rejected frozen-memory candidate unless the exact candidate, code, and split mapping is demonstrated.

### 6.2 External temporal robustness result

The retained aggregate external report identifies the best observed rejected candidate as:

| Field | Recovered value |
|---|---|
| Evaluation status | `external_evaluation_rejected` |
| Candidate | `global_global__coverage_025` |
| Pooled return | -6.017% |
| Pooled Sharpe | -0.376002 |
| Maximum drawdown | -13.591% |
| Positive markets | 2 |
| Positive-profit concentration | 89.596% |
| Dominant positive-profit market | China |

This is external temporal robustness evidence for a candidate that already failed promotion. It is not a confirmation of a promoted or locked model.

## 7. Items only partially recovered or still unavailable

| Credibility-debt area | Current status | What is missing |
|---|---|---|
| Principal checkpoint binaries and hashes | Unavailable | No `.pt`, `.pth`, or `.ckpt` files are in the complete bundle. Checkpoint paths exist in training records, but no binaries or selected-checkpoint hashes do. |
| Optimizer/GradScaler state | Unavailable | No resumable optimizer-state artefacts were retained. |
| Exact generated/resolved configurations | Partial | Input configuration blobs and config fingerprints are present; generated per-job configs are not retained for the principal run. |
| Principal dirty-tree state | Partial | Commit recovery is strong, but the original dirty diff/submodule record is absent. |
| Full principal package lock | Partial | Runtime versions are retained; full package inventory was captured later during evidence export, not demonstrated for the principal run. |
| Raw price/fundamental snapshots | Unavailable | No immutable provider responses, raw market Parquet snapshots, retrieval timestamps, or processed-data hashes are in the bundle. |
| Point-in-time release timing | Unavailable | Intraday earnings/report timestamps and a common cross-market information cutoff are not recoverable from outputs alone. |
| Symbol-level data exclusions | Partial | The five identities are recovered: `CIEL3.SA`, `JBSS3.SA`, `EMBR3.SA`, `STM.PA`, and `AHT.L`. Their technical/data exclusion reasons still require a source data-quality ledger. |
| Representation audit | Unavailable | Principal latent export files, original assignments, ARI calculation records, association estimator, sample rules, and dependence-aware intervals are absent. |
| ADBE audit | Partial | ADBE appears in multiple trade/evaluation ledgers, but the register's specific gross-versus-net example is not yet uniquely linked to the principal rejected candidate. |
| Fresh prospective evaluation | Not repairable by recovery | An already inspected interval cannot become prospective validation through archiving or rerunning. |

## 8. Scientific limitations that remain in force

The following are empirical outcomes or validity limitations, not wording problems:

1. All 12 development candidates failed the promotion decision.
2. No candidate was selected or locked; no original confirmation job was generated.
3. Development gains were geographically concentrated, particularly in China.
4. Nominal reliability coverage was heterogeneous across markets and should not be described as calibrated without a separate validated analysis.
5. The later external evaluation was economically negative and had only two positive markets.
6. The retained evidence does not establish H1, H2, H3, transferable profitable alpha, or a validated offline-RL policy.

## 9. Required reporting boundary

The following claims are supportable after this recovery:

- the historical run plan, source baseline, core implementation, runtime characteristics, promotion contract, candidate leaderboard, and negative selection result;
- the existence of detailed trade/signal/neighbour/metric artefacts for named run families;
- the negative external robustness result for the best observed rejected candidate.

The following claims must remain withheld pending primary artefacts or new work:

- exact checkpoint reproducibility or independent end-to-end rerunning;
- immutable point-in-time data validity and release-time causality;
- representation-stability statistics and identity-leakage controls;
- a fully specified, investable cross-currency pooled portfolio interpretation unless the backtest/data contract is separately reconstructed and audited;
- calibrated reliability filtering, validated simple-method superiority, transferable profitability, or confirmed offline-RL efficacy.

## 10. Authoritative local paths

| Artefact | Path |
|---|---|
| Full reviewer bundle | `C:\Users\rohil\Downloads\core_rl_complete_paper_evidence\manuscript_evidence` |
| Compact principal evidence | `C:\Users\rohil\Downloads\core_rl_paper_evidence (3)\rl` |
| Principal manifest | `C:\Users\rohil\Downloads\core_rl_paper_evidence (3)\rl\original_summaries\experiment_manifest.yaml` |
| Principal contract | `C:\Users\rohil\Downloads\core_rl_complete_paper_evidence\manuscript_evidence\project\reports\final_testbed\phase6_a30_final_v1\orchestration_state\contract.json` |
| AMP migration record | `C:\Users\rohil\Downloads\core_rl_complete_paper_evidence\manuscript_evidence\project\reports\final_testbed\phase6_a30_final_v1\orchestration_state\migrations\0001_source_migration.json` |
| Candidate leaderboard | `C:\Users\rohil\Downloads\core_rl_paper_evidence (3)\rl\original_summaries\leaderboard.csv` |
| Selection outcome | `C:\Users\rohil\Downloads\core_rl_paper_evidence (3)\rl\original_summaries\selection.json` |
| External evaluation report | `C:\Users\rohil\Downloads\core_rl_paper_evidence (3)\rl\original_summaries\confirmation_report.md` |

## 11. Recommended next recovery deliverables

1. Produce an output-to-job index that joins every principal metric, signal, neighbour, and trade artefact to job ID, seed, market, topology, split, source fingerprint, and checkpoint path.
2. Extract a formal methods appendix directly from commit `2392260`, with formulas and exact parameter semantics for features, targets, retrieval, reliability, policy, costs, and pooling.
3. Recompute the complete promotion table from the retained leaderboard/market outputs, showing every Boolean gate failure per candidate.
4. Create a run-family-safe baseline appendix; do not mix later study outputs with the principal rejected candidate.
5. Retrieve from the VM or backup system: checkpoint binaries, optimizer states, raw immutable snapshots, data manifests/hashes, original latent exports, and the dirty-worktree patch if it exists.
6. Run a newly frozen, preregistered prospective evaluation only after the above contracts are locked.

## 12. Resolution addendum: previously partial items

This addendum resolves the parts that can be established from the retained local evidence and isolates the parts that remain unavailable.

### 12.1 Historical source identity — substantially resolved

The principal post-migration source fingerprint matches Git commit `23922607a8d45c47c198fde609f0f046440231f7` in 128 of 132 contracted files. A search through the complete local Git history found no committed blob matching the four remaining helper scripts. They were therefore uncommitted or otherwise outside the retained Git history at the time the contract was written.

This is sufficient to identify the authoritative historical implementation for features, targets, training, retrieval, aggregation, policy, reliability, backtesting, and selection. It is not sufficient to reconstruct the original dirty diff for the four helper scripts. No Gitlink entries or `.gitmodules` entry exist at the recovered commit; the principal run therefore had **no Git submodules** to record.

### 12.2 Launch execution — resolved as plan, not as shell history

The retained manifest is a complete executable DAG, not merely a summary. It identifies 551 jobs, each command argument list, input, dependency, expected output, GPU requirement, and stage. It can therefore reproduce the designed execution plan exactly.

No retained shell history, scheduler transcript, or process accounting log proves that every invocation ran with no unrecorded shell-level change. This distinction must remain in the manuscript.

### 12.3 Generated configurations — partially resolved

The historical input blobs `configs/final_research_testbed.yaml`, `configs/final_encoder_training.yaml`, `configs/phase4e_cross_market_sharpe.yaml`, and `requirements-phase5-rl.txt` match the run contract exactly. Each retained training record also stores a configuration fingerprint.

No per-job generated configuration file is retained for the principal run. The missing files can only be recovered from a VM/backup copy of the run root, or regenerated from commit `2392260` and the manifest; regeneration would demonstrate the intended configuration, not prove that the historical generated file was byte-identical.

### 12.4 Runtime environment — materially resolved, still not a full lock

The 30 principal training-completion records provide observed Python, PyTorch, CUDA, cuDNN, GPU model, compute capability, and VRAM. This establishes the principal accelerator/runtime envelope.

The exact package lock, OS library inventory, container image digest, and environment variables remain absent. They can be recovered only from a VM environment snapshot, an image registry, shell history, or a previously exported package-lock file.

### 12.5 AMP migration — affected job and post-migration artefact resolved

The migration record identifies the only reset job as `encoder_global_seed_7`, whose prior status was `failed`. The retained post-migration training record for `models/global_seed_7` completed at `2026-07-24T23:20:39`, after the recorded migration time of `2026-07-24T23:01:26`.

Thus the affected job and its successful post-migration artefact are identified. What is still unavailable is the pre-migration failed checkpoint/output, so no archival comparison can prove the numerical difference between pre- and post-migration results.

### 12.6 Costs and cross-market pooling — implementation semantics resolved

The historical implementation gives the following precise semantics:

- `market_memory_backtester.py` applies slippage by multiplying buy fills by `1 + bps / 10,000` and sell fills by `1 - bps / 10,000`; its policy default is 10 bps and its initial capital is 100,000 units.
- The historical market configuration declares market execution costs of 10 bps (US), 20 bps (India), 20 bps (China), 15 bps (Brazil), 20 bps (France), and 25 bps (UK). It also declares a 10-bps policy-training transaction-cost setting.
- The principal confirmation aggregation computes each market/seed equity return series, takes the date-wise arithmetic mean with `skipna=True`, then compounds that mean return series. The final-selection script likewise averages available market return series by date; it fills missing candidate returns with zero only when constructing the PBO matrix.
- No FX conversion, multi-currency cash ledger, or closed-market carry-forward valuation is implemented in the recovered aggregation code.

This resolves the code-level pooling contract: it is an equal-weight aggregation of local portfolio return series, not a single fully specified FX-converted capital account. It does **not** resolve which of the configured 10–25 bps schedules was injected into every historical principal evaluation; per-job generated configurations remain required for that claim.

### 12.7 Principal baseline and RL artefacts — mapped at run-family level

The principal run contains an unambiguous path-level mapping for the following output families:

| Family | Recovered coverage | Interpretation |
|---|---|---|
| Principal memory evaluations | 72 metrics and 72 trade ledgers | `global`/`regional` encoder, six markets, seeds 7/17/37, and `policy_development`/`policy_selection` splits. |
| Pilot RL evaluations | 16 metrics and 16 trade ledgers | `bandit`, `cql`, `iql`, and `td3bc` across US, India, China, and Brazil. |
| Full RL evaluations | 72 metrics and 72 trade ledgers | IQL and TD3+BC, global/regional variants, six markets, and seeds 7/17/37. |

The directory structure is sufficient to prevent anonymous-file comparisons. It is not sufficient to claim that later `final_memory_*`, transfer, or temporal-transport artefacts are principal-run baselines; those remain separate run families unless their manifest/code/split equivalence is demonstrated.

### 12.8 ADBE example — partially resolved

ADBE occurs in retained principal and later-family ledgers, so the evidence is not absent. For example, the principal `global_US_seed_7` and `regional_US_seed_*` memory ledgers contain ADBE trades, and the separate `final_memory_policy_v1` US ledger contains five ADBE trades with both gross and net fields.

The specific register example reporting gross return -5.86% versus a P&L-implied -6.05% is not uniquely identifiable from file name, candidate, date, and checkpoint metadata in the retained bundle. It must remain unmapped. The required recovery route is the original diagnostic-run manifest/report, or a ledger index that records the example's run ID, candidate, seed, split, and checkpoint.

### 12.9 Reliability — implementation and statistical interpretation separated

The reliability mechanism is now fully reconstructable from the historical code:

- the binary event is `future_blended_alpha_63 > 0.002` after costs;
- the fit fraction is 70%, with a 63-session calibration embargo;
- minimum fit and calibration sample sizes are 500 and 200 rows, respectively;
- the classifier is regularized logistic regression (`C=0.10`), optionally isotonic-calibrated; a robust Huber error model is also fitted;
- calibration thresholds are quantiles of calibrated reliability probabilities;
- Brier skill is measured against the fitted training base rate; ECE uses 10 quantile bins.

This resolves the semantic/algorithmic debt. It does not establish that the resulting probabilities are well calibrated in every market. The retained market results instead show heterogeneous realised coverage and must be reported as empirical calibration evidence, not inferred from the algorithm.

### 12.10 PBO and DSR reconciliation — resolved

PBO and deflated-Sharpe probability are different statistics and must remain separate:

| Statistic | Recovered value for best observed candidate | Meaning |
|---|---:|---|
| PBO | 0.0 | The selection-program overfitting estimate computed from the candidate return matrix. |
| Deflated-Sharpe probability | 0.4443886586 | The leaderboard's multiple-testing-adjusted Sharpe-related quantity. |

Neither value is a substitute for the other. PBO=0.0 does not override the candidate's failed promotion result, and the 0.4444 deflated-Sharpe probability is not a promotion pass.

### 12.11 Configured gates versus implemented selection — resolved

The two contracts are distinct and should be displayed separately in any manuscript:

| Source | Conditions |
|---|---|
| `final_research_testbed.yaml` | Pooled Sharpe >= 2.0; drawdown <= 20%; at least five positive markets; PBO <= 0.50; profit concentration <= 35%; nondegenerate actions required. |
| `select_final_policy.py` | At least five baseline market wins; at least five nondegenerate markets; worst drawdown <= 20%; median excess Sharpe > 0; PBO <= 0.50. |

For `global_global__coverage_025`, the recovered leaderboard gives pooled Sharpe 1.503026 (below 2.0) and positive-profit concentration 70.2909% (above 35%). The candidate also has only three baseline wins in the retained leaderboard. It therefore fails material configured and implemented selection conditions, irrespective of PBO.
