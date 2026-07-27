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
- The Phase 6 offline-RL comparison selected IQL and TD3+BC at pilot, but every
  final policy/representation candidate failed the six-market development gate.
- Global transformer plus regionally adapted internal and external memory
  reached only about 0.69 pooled Sharpe and was rejected.

These figures are development evidence from ignored local report artifacts.
They are not confirmation evidence and are deliberately absent from a clean
GitHub clone.

## Frozen Candidate

The remaining candidate is fixed in `configs/final_memory_policy.yaml`:

1. Shared global patch Transformer.
2. Global static internal memory.
3. Global external historical memory.
4. Three-seed consensus with the 25% reliability-coverage candidate.
5. Deterministic threshold policy; no RL.

The earlier global/global development sweep reached approximately 1.50 pooled
Sharpe but still failed the complete promotion contract. This is the strongest
remaining architecture, not a claimed profitable strategy. Its structure and
thresholds are frozen to prevent further selection-period overfitting.

## Next Evaluation

The next work is comparison rather than architecture tuning:

1. Reproduce the locked candidate and conventional baselines under one cost and
   chronology contract.
2. Evaluate additional markets or datasets without changing the candidate.
3. Run the untouched temporal confirmation only under a predeclared protocol.
4. Report failed markets, drawdown, turnover, calibration, and profit
   concentration alongside pooled Sharpe.

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
- The Phase 6 international RL testbed was executed and rejected. Its compiler
  remains historical infrastructure, not the active research direction.
