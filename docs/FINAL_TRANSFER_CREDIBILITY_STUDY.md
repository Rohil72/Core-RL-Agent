# Final Transfer Credibility and Mechanism Study

## Research Question

This experiment answers one concrete question:

> Which property of the representation, memory, or market distribution causes
> the strong development Sharpe to disappear under temporal and cross-market
> transfer?

It is a diagnostic study over preserved artifacts. It neither retrains the
Transformer nor reopens the failed sealed confirmation.

## Frozen Inputs

- Global patch Transformer checkpoints from `phase6_a30_final_v1`.
- Global raw 128-dimensional state embeddings.
- Corrected complete-period transfer decisions from `final_memory_repair_v2`.
- Six fixed markets and seeds 7, 17, and 37.
- The original C0 future opportunity, downside, path, and holding outcomes.
- The deterministic top-three, long-only, cost-aware policy.

The 2025 through 2026 Q1 interval has already been observed. It remains a
contaminated diagnostic period and cannot be called confirmation.

## Memory Transport Candidates

All candidates use the same encoder, targets, query states, policy, and costs.
Only the historical evidence transport changes.

| ID | Memory bank | Time rule | Market weighting |
| --- | --- | --- | --- |
| `m0_raw_static` | Frozen pre-2022 | None | Natural frequency |
| `m1_raw_growing` | Causally expanding | None | Natural frequency |
| `m2_raw_growing_half_life_2y` | Causally expanding | 730-day half-life | Equal total evidence per source market |
| `m2_raw_growing_half_life_4y` | Causally expanding | 1461-day half-life | Equal total evidence per source market |
| `m2_raw_growing_hard_window_4y` | Causally expanding | Exclude evidence older than 1461 days | Equal total evidence per source market |

An expanding bank may contain later observations, but a query can retrieve a
row only after its exact outcome-availability timestamp. Recency and market
balance affect evidence weights only; they do not change latent distances.

## Conventional Decoder Baselines

The study also fits three causal decoders on the same frozen raw latent states:

- ElasticNet.
- Histogram gradient-boosted regression trees.
- PCA followed by distance-weighted k-nearest-neighbor regression.

They use only outcomes available before the query period begins and execute
through the same top-k policy and transaction-cost assumptions. These are
frozen-state decoding baselines, not raw-feature technical-analysis baselines.
They test whether memory adds value beyond a conventional function of the same
representation.

## Credibility Outputs

For every period, the audit writes:

- paired circular block-bootstrap intervals at 5, 21, and 63 sessions;
- market-cluster bootstrap comparisons;
- leave-one-country-out pooled Sharpe and annualized return;
- Deflated Sharpe probability and PBO over the declared candidates;
- retrieval reliability deciles against realized 63-session alpha;
- neighbor-age contribution buckets;
- source-market concentration of retrieved evidence;
- memory and conventional-decoder strategy tables.

The primary pooling rule is equal weight across markets on each date. It does
not pool trades or capital in a way that lets a larger market silently dominate.

## Interpretation Matrix

- **M1 beats M0 broadly:** stale static memory is the limiting factor.
- **M2 beats M1 broadly:** temporal drift or source-market imbalance is the
  limiting factor.
- **A conventional decoder beats memory:** neighborhood semantics or evidence
  aggregation is the limiting factor.
- **All frozen-state methods fail in the same markets/years:** the latent state
  or available predictors do not transport enough signal.
- **Good pooled results fail country jackknife:** performance is concentration,
  not market-agnostic transfer.
- **Confidence deciles do not order realized alpha:** the confidence mechanism
  is descriptive rather than decision-calibrated.
- **Only old neighbor buckets carry alpha:** development performance depends on
  a historical regime that no longer transfers.

## Durable VM Execution

The run is CPU-capable and declares zero GPU training or inference jobs because
it reuses the corrected transfer decisions already preserved on the VM.

```bash
cd /home/core-rl-phase6
git pull --ff-only origin main
source .venv/bin/activate
RUN_ID=final_transfer_credibility_v1
```

Build the immutable DAG:

```bash
.venv/bin/python scripts/run_final_memory_study.py \
  --config configs/final_transfer_credibility.yaml \
  --run-id "$RUN_ID" \
  --stage build \
  --python .venv/bin/python
```

Inspect the plan:

```bash
.venv/bin/python scripts/run_durable_experiment.py \
  --manifest "reports/final_transfer_credibility/$RUN_ID/experiment_manifest.yaml" \
  --state-dir "reports/final_transfer_credibility/$RUN_ID/orchestration_state" \
  --project-root /home/core-rl-phase6 \
  --plan-summary
```

The plan must contain 418 jobs, five memory variants, 18 tabular-decoder jobs,
three credibility audits, and zero GPU jobs.

Run or resume with the same command after removing `--plan-summary`:

```bash
.venv/bin/python scripts/run_durable_experiment.py \
  --manifest "reports/final_transfer_credibility/$RUN_ID/experiment_manifest.yaml" \
  --state-dir "reports/final_transfer_credibility/$RUN_ID/orchestration_state" \
  --project-root /home/core-rl-phase6
```

Completed jobs remain complete across pauses. The durable runner retries only
missing or failed artifacts.

## Claim Boundary

This experiment can identify why transfer failed and select a mechanism worthy
of a future preregistered test. It cannot produce fresh confirmation because
all three periods and the fixed universes have already influenced research
decisions.
