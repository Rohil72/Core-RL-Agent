# Temporal Decision Loss Study

## Purpose

This is a diagnostic frozen-latent study. It does not retrain the transformer,
alter the memory database, unlock confirmation, or promote a strategy. It asks
why useful development performance disappeared under temporal and cross-market
transfer:

1. Did the adapter learn the wrong temporal interpretation?
2. Did it identify useful outcomes but allocate capital incorrectly?
3. Did its predictive risk vary excessively across market, year, and volatility
   environments?

## Implemented objective

The existing quantile, same-date ranking, analogue geometry, and VICReg losses
remain the baseline. New terms are independently configurable:

- `temporal_event`: multiclass likelihood over no event, upside-first time
  bucket, and drawdown-first time bucket.
- `temporal_ranking`: concordance between earlier observed events and examples
  that remained event-free for longer.
- `temporal_coherence`: state-distance-gated consistency across adjacent
  sessions for the same ticker.
- `opportunity`: date-level stock-versus-cash allocation supervision. Positive
  mature utility creates investable targets; cash is the target when no
  positive candidate exists.
- `coverage`: matches exposure to the number of positive opportunities up to
  the configured portfolio capacity. It prevents reject-all without forcing
  trades on dates with no positive candidate.
- `environment`: variance of predictive risk across market-year-volatility
  environments after trimming a declared fraction of extreme losses.

The event head is parameterized as one probability distribution over competing
event and time classes. Cumulative upside and drawdown probabilities are sums of
those class probabilities, so they are monotonic by construction. The optional
utility parameterization similarly guarantees `q10 <= q50 <= q90`.

## Temporal labels

Event labels are built from future prices only for mature training examples.
The default study uses horizons `5, 10, 21, 42, 63` and ex-ante trailing
volatility barriers. Barrier volatility uses prices available at the query
timestamp; future prices only determine the realized event class and time.

Each label is one of:

```text
0                      no event by 63 sessions
1 .. H                 upside first, in horizon bucket
H + 1 .. 2H            drawdown first, in horizon bucket
```

## Factorial ablation

`configs/temporal_decision_ablation.yaml` declares five variants:

| Variant | Temporal | Opportunity | Robustness |
| --- | ---: | ---: | ---: |
| `baseline` | No | No | No |
| `temporal_only` | Yes | No | No |
| `action_only` | No | Yes | No |
| `temporal_action` | Yes | Yes | No |
| `temporal_action_robust` | Yes | Yes | Yes |

All variants reuse frozen latent exports, identical chronological splits, the
same backtester, and the same retrieval-based checkpoint-selection rule. The
study writes per-run temporal Brier score, event accuracy, event-time bucket
MAE, monotonicity violations, opportunity regret, coverage error, retrieval
quality, and trading metrics.

The active configuration discovers the preserved Phase 6 encoder exports under
`reports/final_testbed/phase6_a30_final_v1/latents`. Regional sources use only
their matching `data/international/MARKET` price directory. Global sources use
the all-market price pool. The older Phase 4C source contract remains available
through `experiment.source_mode: phase4c`.

## Interpretation contract

- Temporal improvement without trading improvement means the remaining weak
  link is allocation or memory evidence, not event semantics.
- Action improvement without temporal improvement means the latent signal was
  usable but previously misallocated.
- A robustness improvement must raise held-out market/year stability without
  erasing mean utility or exposure.
- Lower drawdown caused by collapsed exposure is not an improvement.
- No result from this study is untouched confirmation because its source
  periods and universes have already been inspected.

## Execution

Verify source artifacts before starting training:

```bash
.venv/bin/python scripts/run_temporal_decision_ablation.py \
  --config configs/temporal_decision_ablation.yaml \
  --run-id temporal_decision_pilot_v1 \
  --max-runs 1 \
  --preflight
```

One-run smoke test:

```bash
.venv/bin/python scripts/run_temporal_decision_ablation.py \
  --config configs/temporal_decision_ablation.yaml \
  --run-id temporal_decision_smoke_v1 \
  --variants temporal_action_robust \
  --max-runs 1 \
  --epochs 1 \
  --skip-backtest
```

Complete diagnostic:

```bash
.venv/bin/python scripts/run_temporal_decision_ablation.py \
  --config configs/temporal_decision_ablation.yaml \
  --run-id temporal_decision_ablation_v1
```

Resume the same run with `--resume`. Use `--variants` to run or resume only a
declared subset.

## Research basis

- Lee et al., [DeepHit: A Deep Learning Approach to Survival Analysis With
  Competing Risks](https://doi.org/10.1609/aaai.v32i1.11842), AAAI 2018.
- Brando et al., [Deep Non-crossing Quantiles through the Partial
  Derivative](https://proceedings.mlr.press/v151/brando22a.html), AISTATS 2022.
- Mandi et al., [Decision-Focused Learning: Through the Lens of Learning to
  Rank](https://arxiv.org/abs/2112.03609), ICML 2022.
- Geifman and El-Yaniv, [SelectiveNet: A Deep Neural Network with an Integrated
  Reject Option](https://proceedings.mlr.press/v97/geifman19a.html), ICML 2019.
- Krueger et al., [Out-of-Distribution Generalization via Risk
  Extrapolation](https://proceedings.mlr.press/v139/krueger21a.html), ICML 2021.
- Zhai et al., [DORO: Distributional and Outlier Robust
  Optimization](https://proceedings.mlr.press/v139/zhai21a.html), ICML 2021.
- Zhao et al., [On Learning Invariant Representations for Domain
  Adaptation](https://proceedings.mlr.press/v97/zhao19a.html), ICML 2019.
