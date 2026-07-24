# Research Status

## Established Direction

The project has moved from weak detector labels to representation learning for
future opportunity and risk. The active hypothesis is that frozen latent states
plus causal retrieval can provide more useful trading evidence than a direct
prediction head.

Existing local development artifacts support continuing this line of work, but
they do not establish a promoted strategy. Retrieval and confidence-aware memory
looked more useful than the direct model-head policy, while results varied by
fold and seed.

## Rejected Development Hypotheses

The following local Phase 5 runs were rejected by their own gates:

- Adapter promotion variants: pooled out-of-fold Sharpe about 1.65 and 1.72.
- Consensus selector: PBO about 0.53.
- Consensus exposure: leave-one-fold-out Sharpe about 2.15 and PBO about 0.07,
  but it did not pass the complete promotion contract.
- Final opportunity allocator: improved several folds but had no out-of-year
  positive folds under its configured criterion.

These figures are development evidence from ignored local `reports/phase5/`
artifacts. They are not confirmation evidence and are deliberately absent from
a clean GitHub clone.

## Current Question

Can one predeclared offline-policy algorithm turn a frozen regional or global
encoder plus regional historical memory into consistent exposure decisions across
different local markets?

The executable answer is fixed in `configs/final_research_testbed.yaml`:

1. Audit six market datasets.
2. Pilot bandit, CQL, IQL, and TD3+BC on US, India, China, and Brazil with seed 7.
3. Select exactly two algorithms, then compare regional/global encoders across
   six markets and seeds 7, 17, and 37.
4. Promote at most one candidate only if all development gates pass.
5. Lock source, data, runtime, models, and datasets; evaluate the untouched
   candidate on 2025 through 2026 Q1.

## Promotion Standard

Development requires positive median excess Sharpe versus the memory baseline,
non-degenerate exposure and wins in at least five markets, maximum drawdown no
greater than 20%, and PBO no greater than 0.50. Locked confirmation further
requires pooled Sharpe at least 2.0, five positive markets, drawdown at most
20%, and limited profit concentration.

## Known Limits

- Fixed liquid-company universes retain survivorship and selection bias; they
  are reproducible test universes, not historical index constituent sets.
- International fundamental coverage differs by market. The testbed permits
  zero minimum coverage, so coverage must be audited before interpreting a
  result as fundamental-aware transfer.
- Each market declares a benchmark, but decision data falls back to local
  cross-sectional median return when benchmark alpha is unavailable.
- The final international DAG has been compiled and tested, but no final
  encoder or offline-RL training has been launched from it yet.
