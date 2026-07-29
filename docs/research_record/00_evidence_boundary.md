# Evidence Boundary

## What Counts as a Result

This record separates four things that were repeatedly confused during the
research:

1. A latent-space diagnostic can show representation structure without proving
   profitable trading.
2. A development Sharpe can show a promising mechanism without being fresh
   confirmation.
3. A selected candidate can still fail a later cross-market or confirmation
   gate.
4. A run can complete technically while remaining invalid because of missing
   data, lookahead, wrong embedding columns, or an uncalibrated score.

One later audit found a mixed timestamp-precision lookahead defect: Parquet
availability timestamps could be microsecond-resolution while query timestamps
were compared as nanoseconds. Earlier pre-fix causal-memory results are kept as
historical diagnostics. Only artifacts rebuilt after explicit nanosecond
normalization and a zero-violation audit can support causal claims.

The project uses development, selection, and observed diagnostic periods. The
2025 through 2026 Q1 period has been inspected and is not fresh confirmation.
Fixed liquid-company universes retain survivorship and selection bias.

## Shared Architecture

The research evolved into:

```text
technical and fundamental observations
  -> patch Transformer encoder
  -> 128-dimensional latent state
  -> optional 128 -> 64 -> 32 decision adapter
  -> causal historical market memory
  -> evidence aggregation and opportunity score
  -> deterministic long-only policy
```

Offline RL was tested later, but the full international comparison rejected it.
That experiment evaluated per-seed, one-day-reward allocators rather than the
earlier seed-consensus policy. It therefore rejects that RL formulation, not
every possible RL extension of the consensus-memory system.

## Promotion Standard

The declared target was pooled Sharpe at least 2.0, at least five positive
markets, maximum drawdown at most 20%, controlled concentration, stable seed
lift, and multiple-testing-aware gates such as PBO and deflated Sharpe. Passing
one internal period or one market never satisfied that standard.

The strongest development-only C0 result, pooled Sharpe about 2.15, came from
the static median-consensus policy. It must not be credited to the later
exposure controllers, which consistently reduced that development performance.
