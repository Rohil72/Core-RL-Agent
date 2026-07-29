# Final Enriched Memory Study

`[REJECTED] [NEEDS-REPAIR]`

## Contract

The run used existing regional encoders and adapters only. It created 322 CPU
jobs with no Transformer, adapter, or RL training. It compared:

1. `raw_c0_static`: raw 128D C0 memory.
2. `adapter_c0_static`: adapter 32D C0 memory.
3. `adapter_rally_balanced_static`: rally-path targets, empirical tails, and
   ticker-balanced retrieval.
4. `adapter_rally_multiscale_growing`: causal growing memory with 10, 25, and
   50-neighbor evidence agreement.

The bundle contains 18 view summaries, 72 evaluation metrics, 72 consensus
signal parquet files, 12 aggregate summaries, and 3 comparison summaries.

## Aggregate Results

| Variant | Development Sharpe | Selection Sharpe | Observed Sharpe |
|---|---:|---:|---:|
| Raw C0 | -0.157 | -0.037 | 0.322 |
| Adapter C0 | 0.387 | -0.586 | -0.471 |
| Rally static | 0.415 | 0.412 | 0.116 |
| Rally multiscale | 0.409 | 0.450 | 0.351 |

The best single observed-period variant, multiscale rally memory, returned
3.77% with pooled Sharpe `0.351`. The 2+ Sharpe result appeared only for US
selection at `2.422`; US fell to `0.960` observed.

## What Worked

- Rally-aligned outcomes were much better than adapter C0 semantics.
- Multiscale memory was the most stable enriched variant across periods.
- Stop-loss trades fell from 40 to 14 in selection and from 65 to 28 observed
  when moving from static rally memory to multiscale growing memory.
- Multiscale memory beat equal-weight in 4/6 development markets, 4/6
  selection markets, and 2/6 observed markets.
- Across all markets, the top 20% of multiscale rank had higher realized alpha
  than the bottom 20% in development, selection, and observed periods.

## Why It Was Rejected

- Positive markets were 3/6 in development and selection, then 2/6 observed.
- Median market Sharpe was negative in the observed period for every variant.
- France and UK frequently had negative top-ranked alpha, showing that the
  rank is not market-agnostic.
- Confidence and scale agreement were not reliably predictive of future alpha.
- Pooled drawdown hid severe single-market drawdowns. Multiscale worst-market
  drawdown was approximately -46.3% development, -33.9% selection, and -36.2%
  observed.

## Two Validity Defects

### Unbounded Path Quality

`decision_path_quality` is defined as:

```text
MFE / (abs(MAE) + 1e-6)
```

It can reach tens of thousands when adverse excursion is close to zero. The
enriched score adds this quantity as though it were a bounded probability,
creating opportunity scores above 100 even though returns are fractional. The
current rally ranking is therefore not a clean test of the intended alpha-tail
score.

### Incomplete Observed Inference

India's observed-period predictions end on 18 March 2025 while its price and
outcome rows continue through 31 March 2026. Only 17.5% of India observed rows
have usable predictions. China and Brazil also have partial coverage in some
periods. This must be repaired before calling the international comparison
complete.

## Correct Interpretation

The final study supports a useful but narrower claim: richer historical memory
can improve opportunity ranking and reduce catastrophic entries, but the
current evidence does not establish a robust profitable strategy across
markets. The next legitimate test is a surgical repair of path-quality scaling,
complete inference coverage, and per-market drawdown/coverage gates, followed
by only the two rally variants.

## Closing Repair Protocol

The repair implementation is frozen in `configs/final_memory_repair.yaml`. It
regenerates complete global-model inference, bounds path quality to `[0, 1]`,
enforces nanosecond causal timestamps, and compares only raw C0, bounded
rally-static, and bounded rally-multiscale memory. It remains diagnostic because
every available period has already been inspected.

## Source Map

- `configs/final_memory_study.yaml`
- `configs/final_memory_repair.yaml`
- `scripts/run_final_memory_study.py`
- `docs/FINAL_MEMORY_STUDY.md`
- `docs/FINAL_MEMORY_REPAIR.md`
- `reports/final_memory_study/final_memory_study_v1/`
