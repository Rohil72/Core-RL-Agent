# Phase 4A-C: Encoder Stability, Losses, and Rotating Holdouts

`[PARTIAL] [POTENTIALLY-CONTAMINATED]`

## Encoder and Loss Experiments

Phase 4 tested whether the representation could be made more useful through
outcome geometry, analogue-aware losses, and stronger retrieval-oriented
objectives. The Transformer architecture, feature pipeline, detector, oracle,
and dataset were treated as frozen during the later memory-focused branches.

The loss sweep included outcome regression, ranking, analogue geometry, and
VICReg-style terms. The relevant lesson was not a single winning loss; it was
that latent usefulness had to be measured by analogue retrieval and trading
utility rather than validation loss alone.

## Numerical Stability Finding

The `huber_analogue_v2` training run exposed instability in
`F.normalize(latent, p=2, dim=1)`. Very small latent norms caused excessive
backpropagated gradients. The normalization epsilon was increased to
the equivalent of a normalization floor near `1e-4`, bounding the worst scale
of the normalization gradient and resolving the observed failure without
materially changing ordinary latent distances.

This was a training-stability repair, not evidence of trading improvement.

## Phase 4C Holdout/Data Findings

The first five-fold manifest incorrectly contained absent tickers such as `SQ`
and `VZIO`; it was repaired to cover the actual 28-ticker dataset exactly once.
The final run produced 14 of 15 requested fold/seed runs. All completed
retrieval audits passed, but one seed remained non-finite.

| Aggregate result | Value |
|---|---:|
| Median fold return | +10.7% |
| Median fold Sharpe | 0.554 |
| Median excess versus equal weight | -5.26% |
| Mean weekly Rank IC | +0.022 |
| Promotable | No |

Four folds were profitable, but every fold trailed equal weight. Fold 2 was a
large negative outlier. Phase 4C therefore established causal evaluation
machinery and heterogeneous ticker-universe performance, not transferable
alpha.

The later timestamp-precision audit means the original Phase 4C causal
artifacts should not be used as clean promotion evidence unless regenerated
under the repaired timestamp contract.

## Source Map

- `scripts/run_phase4_loss_sweep.py`
- `src/losses/outcome_geometry.py`
- `configs/phase4c_rotating_holdouts.yaml`
- `scripts/run_phase4c_rotating_holdouts.py`
- `docs/PHASE4D_RETRIEVAL_SEMANTICS.md`
- `reports/phase4c/phase4c_v1/aggregate_summary.md`
