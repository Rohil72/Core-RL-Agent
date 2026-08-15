# Temporal-Transport Encoder Study

`[PARTIAL] [NEEDS-REPAIR] [DIAGNOSTIC-ONLY]`

## Direct Verdict

The temporal-transport objective produced a consistent improvement in the
economic quality and cross-ticker diversity of retrieved neighbours. This is
the strongest evidence from the study and is usable as a representation and
retrieval result.

The emitted trading-performance verdict is not interpretable. The pooled
international backtest valued an open position at zero whenever that
security's market was closed while another market was open. The resulting
calendar artefact produced drawdowns near `-100%`, infinite Sortino ratios,
and positive Sharpe ratios for some negative-return policies. The six trained
encoders and twelve saved signal tables remain valid inputs for a corrected
CPU-only re-evaluation; transformer retraining is not required.

The 2025 through 2026 Q1 interval was already observed before this study and
is diagnostic only. It is not a fresh confirmation set.

## Supplied Evidence

The reviewed temporal-only archive was:

`temporal_transport_v2_reviewer_evidence/manuscript_evidence/project/reports/temporal_transport_encoder/temporal_transport_v2`

It contained 80 files totalling approximately 462 MB:

- one experiment manifest and build summary;
- six resolved encoder configurations;
- twelve resolved memory-policy configurations;
- six `training_complete.json` records;
- twelve metrics files, trade ledgers, signal tables, and neighbour tables;
- run-level and paired comparison outputs;
- the durable contract and audited source/runtime migration record.

The build declared six GPU training jobs and twelve evaluation jobs. All six
completion records were present. The manifest SHA-256 was
`0ce0932e9a23d0a459a14ba569192ddd312a1fb53059f7c4375758e3d31724db`.
The final source fingerprint was
`9d897f187c201606bbf861283ef50ccfc0a7156d27db0f6ebedc77970b129b13`.

The recorded runtime was Python 3.10.20, PyTorch 2.2.2 with CUDA 12.1,
cuDNN 8902, and one NVIDIA A30 with 25,337,004,032 bytes of VRAM. The audited
migration changed only the Linux kernel release and three operational runner
files; model, data, loss, and experiment configuration were unchanged.

## Experimental Contract

All variants used the same global six-market data, model architecture,
optimizer, memory subsystem, deterministic policy, dates, and seeds. The only
intended difference was the training loss.

| Contract item | Value |
|---|---|
| Markets | US, India, China, Brazil, France, UK |
| Seeds | 7, 17, 37 |
| Training | 2013-01-01 through 2022-12-31 |
| Validation | 2023-01-01 through 2023-12-31 |
| Development test | 2024-01-01 through 2024-12-31 |
| Observed diagnostic | 2025-01-01 through 2026-03-31 |
| Encoder | Patch Transformer, `d_model=96`, four heads, four layers |
| Dropout | 0.1 |
| Runtime memory | Static parameter, update rate 0.0 |
| Epochs and batch | 20 epochs, batch size 128 |
| Optimizer settings | Learning rate 0.0002, weight decay 0.0001, gradient clip 1.0 |
| Precision | Automatic AMP with FP32 fallback |
| Memory | Gaussian aggregation, 25 neighbours, same ticker excluded |
| Policy | Top three, 21-session minimum hold, 63-session maximum hold |
| Costs | 10 bps configured slippage |

The baseline used masked regression only:

```text
lambda_reg       = 1.00
lambda_analogue  = 0.00
lambda_transport = 0.00
lambda_var       = 0.00
```

The candidate added cross-market and cross-period outcome geometry:

```text
lambda_reg        = 1.00
lambda_analogue   = 0.15
lambda_transport  = 0.10
lambda_var        = 0.01
minimum_year_gap  = 2
positive_quantile = 0.25
negative_quantile = 0.75
transport_margin  = 0.25
variance_target   = 0.20
```

The analogue targets were blended 63-session alpha, 63-session maximum
return, 63-session minimum return, and 126-session upside-before-drawdown path
quality, with weights `1.0`, `0.75`, `1.0`, and `0.5` respectively.

## Causal Retrieval Audit

The twelve neighbour tables contained approximately 8.65 million rows. A
direct audit of query, neighbour, and neighbour-outcome-availability
timestamps found:

- future-neighbour violations: `0`;
- outcome-availability violations: `0`;
- latest memory date: `2022-12-30`;
- earliest development query: `2024-01-01`;
- earliest observed query: `2025-01-01`.

Although the generated memory configuration set
`require_outcome_availability: false`, the chronological split itself left all
retrieved training outcomes mature before either query period. This run's
retrieval evidence is therefore causally valid under the recorded split.

## Retrieval Results

### Mean results across three seeds

| Period | Variant | Alpha MAE | Downside MAE | Alpha Spearman | Weekly alpha IC | Top-decile precision | NDCG@25 | Interval coverage | Ticker HHI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | Baseline | 0.1169 | 0.0770 | -0.0225 | -0.0241 | 0.4294 | 0.4991 | 0.1432 | 0.4387 |
| 2024 | Temporal transport | **0.1113** | **0.0727** | **0.0115** | **0.0131** | **0.4431** | **0.5529** | **0.1827** | **0.2873** |
| 2025-Q1 2026 | Baseline | 0.1350 | 0.0850 | -0.0290 | -0.0265 | 0.4377 | 0.4762 | 0.1352 | 0.4294 |
| 2025-Q1 2026 | Temporal transport | **0.1260** | **0.0800** | **-0.0032** | **-0.0024** | **0.4592** | **0.5298** | **0.1727** | **0.2879** |

Lower MAE and HHI are better; higher ranking, precision, NDCG, and coverage
values are better. The observed-period correlations remain near zero, so the
result should be described as a reduction in retrieval error and improved
ordering, not strong predictive correlation.

### Seed-period consistency

Across the six paired seed-period comparisons:

| Diagnostic | Candidate wins |
|---|---:|
| NDCG@25 | 6/6 |
| Lower neighbour ticker concentration | 6/6 |
| Lower alpha MAE | 5/6 |
| Lower downside MAE | 5/6 |
| Better alpha Spearman | 5/6 |
| Better weekly alpha IC | 5/6 |
| Better top-decile precision | 4/6 |

This consistency is more informative than a single pooled score. The loss
made neighbourhoods more outcome coherent and less dominated by a small set
of tickers across both periods.

## Direct-Head Diagnostics

The candidate also reduced mean direct future-target error, but did not create
a consistently useful direct alpha correlation.

| Period | Variant | All-target MAE | All-target R2 | Blended-alpha MAE | Blended-alpha Pearson |
|---|---|---:|---:|---:|---:|
| Validation | Baseline | 0.2207 | -0.2880 | 0.1025 | 0.0956 |
| Validation | Temporal transport | 0.2229 | -0.1191 | 0.0958 | 0.0957 |
| 2024 | Baseline | 0.2491 | -0.8882 | 0.1139 | 0.0209 |
| 2024 | Temporal transport | 0.2363 | -0.3784 | 0.1078 | -0.0429 |
| 2025-Q1 2026 | Baseline | 0.2764 | -1.0740 | 0.1336 | -0.0480 |
| 2025-Q1 2026 | Temporal transport | 0.2552 | -0.3622 | 0.1188 | -0.0041 |

The direct head remains weak. The improvement is principally useful as
retrieval geometry, which supports the project's external-memory design rather
than a return to feed-forward prediction.

## Invalid Trading Verdict

The emitted comparison rejected the candidate with one 2024 seed win, mean
2024 Sharpe delta `-0.408`, and mean observed Sharpe delta `-0.696`. These
values must not be cited as valid trading results.

The metrics include the following impossible or incoherent combinations:

- maximum drawdowns between approximately `-96.9%` and `-100%` despite modest
  positive ending returns;
- negative total returns accompanied by positive Sharpe ratios;
- infinite Sortino ratios;
- a candidate with positive return but Sharpe reported as zero.

### Identified mechanism

`run_long_only_backtest` iterates over the union of all international trading
dates. `_positions_value` includes a position only when its ticker exists in
the current date's rows. If India is open while France is closed, for example,
an open French position is temporarily omitted from portfolio value rather
than carried at its last valid close. At high exposure, this produces an
artificial near-total equity loss followed by an artificial recovery when the
security reappears.

This defect affects pooled international equity, daily return, Sharpe,
Sortino, drawdown, Calmar, and any gate derived from them. It does not alter
the saved embeddings, neighbour identities, neighbour outcomes, retrieval
MAEs, NDCG, rank correlations, or concentration diagnostics.

## Required Re-evaluation

1. Preserve the six completed models and twelve signal/neighbour tables.
2. Carry each open position at its last valid close on a market holiday; never
   execute an order on a closed market.
3. Preferably compute one equity curve per market and combine synchronized,
   explicitly currency-treated market returns for the pooled result.
4. Add invariants rejecting non-positive equity, unexplained one-day collapse,
   infinite ratios, and inconsistencies between ending return and daily-return
   compounding.
5. Rerun baseline, temporal transport, momentum, random, and equal-weight
   references from the same saved signal tables.
6. Recreate the comparison and verdict under a new evaluation run ID. Do not
   overwrite the historical invalid verdict.
7. Report 2024 as development evidence and 2025 through Q1 2026 as an observed
   diagnostic only.

This repair is CPU-only and does not justify changing the encoder, loss,
memory, policy thresholds, or training data.

## Publication Interpretation

The currently defensible result is:

> Cross-market, cross-period outcome geometry consistently improved causal
> historical-neighbour quality, outcome ordering, and ticker diversity across
> three seeds and two periods, while a pooled-calendar accounting defect
> prevented valid assessment of downstream portfolio utility.

Do not claim that temporal transport improved or degraded Sharpe until the
saved signals are re-evaluated. Even after repair, the observed period remains
diagnostic and cannot establish untouched temporal confirmation.

## Artifact Map

- Contract: `reports/temporal_transport_encoder/temporal_transport_v2/experiment_manifest.yaml`
- Resolved configs: `reports/temporal_transport_encoder/temporal_transport_v2/generated_configs/`
- Training evidence: `reports/temporal_transport_encoder/temporal_transport_v2/models/*/training_complete.json`
- Retrieval and trade evidence: `reports/temporal_transport_encoder/temporal_transport_v2/memory/`
- Historical emitted verdict: `reports/temporal_transport_encoder/temporal_transport_v2/comparison/verdict.json`
- Study implementation: `scripts/run_temporal_transport_study.py`
- Study configuration: `configs/temporal_transport_study.yaml`
- Backtester requiring repair: `src/backtest/market_memory_backtester.py`
