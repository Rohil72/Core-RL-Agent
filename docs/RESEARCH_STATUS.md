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
- The sealed global/global reliability candidate also failed its observed
  2025-2026 Q1 evaluation. Raw memory remained positive but weak, while the
  reliability gate made the result worse. This closed the reliability-filter
  branch rather than supporting a profitable-strategy claim.

These figures are development evidence from ignored local report artifacts.
They are not confirmation evidence and are deliberately absent from a clean
GitHub clone.

## Rejected Local-Rank Reconstruction

The failed confirmation artifacts and `configs/final_memory_policy.yaml` remain
frozen. The subsequent development hypothesis was separately fixed in
`configs/local_rank_ensemble.yaml`:

1. Reuse each market's existing regional patch Transformer checkpoint.
2. Train only a market-local 128-to-64-to-32 decision adapter.
3. Build an expanding local memory whose outcomes are unavailable until their
   exact maturity timestamp.
4. Rank each market-date independently in three seed spaces.
5. Require zero binary seed votes and apply no reliability filter.
6. Execute the deterministic top-k policy; no RL.

The six-market run did not show robust transfer. Its three-seed ensemble had
negative lift in all three periods and only Brazil was positive in the observed
period. Memory commonly beat the direct adapter head, but did not consistently
beat equal-weight. Audit then found that retrieval selected the raw `latent_*`
columns from mixed raw-plus-adapter frames, so the run did not actually test
adapter-space memory.

## Final Transfer Diagnosis

The closing repair has completed and did not establish broad, stable transfer.
The active work is one bounded causal diagnosis over the preserved artifacts:

1. Reuse corrected global raw embeddings without training any model.
2. Compare static memory with causally growing memory.
3. Add predeclared two-year decay, four-year decay, and a hard four-year window.
4. Balance historical evidence across source markets.
5. Compare memory with ElasticNet, histogram gradient boosting, and PCA-kNN on
   the same frozen states and deterministic policy.
6. Report block-bootstrap intervals, country jackknives, DSR/PBO, reliability
   deciles, neighbor ages, and source-market concentration.

The executable protocol is `scripts/run_final_memory_study.py` with
`configs/final_transfer_credibility.yaml`; see
[Final Transfer Credibility Study](FINAL_TRANSFER_CREDIBILITY_STUDY.md). The 2025-2026 Q1 period has already
been inspected and is diagnostic only. The study cannot promote a strategy,
regardless of its result.

## Active Temporal Decision Diagnosis

The remaining bounded development experiment keeps the transformer and source
latents frozen and tests whether transfer failure arose in temporal semantics
or action inference. It adds competing-risk event timing, monotonic horizon
predictions, stock-versus-cash opportunity regret, opportunity-conditioned
coverage, and trimmed environment-risk variance as independent ablations. See
[Temporal Decision Loss Study](TEMPORAL_DECISION_LOSS_STUDY.md). This study is
diagnostic only and cannot repair the absence of untouched confirmation data.

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
- The six fixed liquid-company universes and all already-inspected periods are
  development evidence. Publication-quality validation still requires an
  untouched universe or later time period fixed before results are viewed.
