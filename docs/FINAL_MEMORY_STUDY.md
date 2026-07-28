# Final Memory Study

## Purpose

`final_memory_study` is the last bounded development experiment on the existing
regional encoders and decision adapters. It trains no Transformer, adapter, or
RL policy. Its only question is whether retrieval becomes economically useful
when the memory representation, outcomes, and evidence aggregation are aligned.

The preceding `local_rank_ensemble_v1` run rejected its tested construction:
the ensemble did not improve on its best seed, only Brazil was positive in the
observed period, and memory did not consistently beat equal-weight. Audit also
found that its memory bank contained both `latent_*` and `decision_*` columns.
The generic memory loader selected `latent_*`, so retrieval happened in the raw
128-dimensional encoder space even though the direct-head comparison used the
trained 32-dimensional adapter space.

This study fixes that ambiguity structurally. Every retrieval view contains
exactly one embedding family named `latent_*`, and its dimension is checked
before execution.

## Fixed Comparison

The four variants are predeclared in `configs/final_memory_study.yaml`:

1. `raw_c0_static`: raw 128-dimensional encoder states with the original C0
   memory targets and static training memory.
2. `adapter_c0_static`: the same C0 targets in the trained 32-dimensional
   decision-adapter space. This isolates embedding alignment.
3. `adapter_rally_balanced_static`: adapter retrieval with maximum favorable
   excursion, maximum adverse excursion, path quality, empirical downside
   tails, and a cap of three neighbors per ticker.
4. `adapter_rally_multiscale_growing`: the richer rally memory with causal
   expanding history and evidence calculated at 10, 25, and 50 neighbors.

The multiscale score is the median opportunity score across the declared
neighborhoods minus a penalty for disagreement. Confidence and agreement are
also reduced when the neighborhood scales disagree on the sign of the
opportunity. This tests whether a rally signal is locally sharp but historically
stable, rather than trusting one arbitrary value of `k`.

Ticker caps prevent one repeatedly sampled company from impersonating many
independent historical analogues. The predictive-tail score uses the empirical
10th percentile of retrieved outcomes and is not presented as a formal
conformal prediction interval.

## Evidence Boundary

- Markets: US, India, China, Brazil, France, and UK.
- Seeds: 7, 17, and 37.
- Development: 2022-2023.
- Selection: 2024.
- Observed diagnostic: 2025 through 2026 Q1.
- Rank aggregation: the original mean seed-percentile rank.
- Policy: the existing deterministic top-three policy; no RL.
- Promotion: forbidden. Every period has already influenced the research.

Where `future_blended_alpha_63` was not stored in the international exports,
the runner derives `decision_return_63 - decision_benchmark_return`. This is
recorded as `derived_local_cross_sectional_alpha`; it must not be described as
sector-adjusted or globally benchmarked alpha.

Interpret the experiment through:

- pooled and per-market Sharpe, return, and maximum drawdown;
- excess Sharpe over equal-weight and momentum;
- lift over each seed and over `raw_c0_static`;
- profit concentration and positive-market count;
- retrieval sign agreement, score dispersion, ticker concentration, and
  neighbor coverage.

No aggregate result can conceal a failed market. The 2.0 Sharpe value is an
aspirational reporting threshold, not a tuning target or proof of validity.

## Execution

Restore the ignored `local_rank_ensemble_v1` artifacts before building.

```bash
RUN_ID=final_memory_study_v1

.venv/bin/python scripts/run_final_memory_study.py \
  --config configs/final_memory_study.yaml \
  --run-id "$RUN_ID" \
  --stage build \
  --python .venv/bin/python

MANIFEST="reports/final_memory_study/$RUN_ID/experiment_manifest.yaml"
STATE="reports/final_memory_study/$RUN_ID/orchestration_state"

.venv/bin/python scripts/run_durable_experiment.py \
  --manifest "$MANIFEST" \
  --state-dir "$STATE" \
  --stage full \
  --plan-summary

.venv/bin/python scripts/run_durable_experiment.py \
  --manifest "$MANIFEST" \
  --state-dir "$STATE" \
  --stage full
```

The manifest has 322 CPU jobs and zero training or RL jobs. Reissuing the final
command validates completed artifacts and resumes the remaining DAG.

## Research Basis

The multiscale and predictive-tail variant is motivated by retrieval-augmented
time-series work and sequential uncertainty estimation, while remaining a
deliberately simpler empirical test:

- Xu and Xie, *Conformal Prediction for Time Series*, ICML 2021:
  https://proceedings.mlr.press/v139/xu21h.html
- Zaffran et al., *Adaptive Conformal Predictions for Time Series*, ICML 2022:
  https://proceedings.mlr.press/v162/zaffran22a.html
- Xu and Xie, *Sequential Predictive Conformal Inference for Time Series*,
  ICML 2023: https://proceedings.mlr.press/v202/xu23r.html
- *Retrieval Augmented Time Series Forecasting* (RAFT), 2025:
  https://arxiv.org/abs/2505.04163
- *Retrieval-Augmented Forecasting* (RAF), 2024:
  https://arxiv.org/abs/2411.08249

These papers support adapting uncertainty to sequential and nonstationary data.
The retrieval preprints also motivate retaining analogue evidence instead of
compressing history into a single parametric prediction. None of them makes the
empirical neighbor percentile in this study conformal or guarantees calibrated
coverage.
