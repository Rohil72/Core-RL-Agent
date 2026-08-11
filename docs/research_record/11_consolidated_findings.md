# Consolidated Findings and Open Gaps

`[CURRENT RESEARCH POSITION]`

## Findings That Survived the Ladder

1. The patch Transformer learns highly reproducible latent states with strong
   future-outcome separation.
2. Those states are not fully ticker-invariant; ticker identity and company
   structure remain entangled.
3. Historical memory can contain more useful trading evidence than a direct
   feed-forward head, but the advantage is weak and inconsistent across
   markets.
4. Outcome semantics matter. Rally-aligned MFE/MAE/path targets outperform a
   generic fixed-horizon adapter target in the final comparison.
5. Causal growing memory and multi-scale agreement reduced stop-loss entries in
   the final study. This is a risk-mechanism result, not established seed lift
   or market-agnostic alpha.
6. The tested per-seed offline-RL formulation did not improve the system
   reliably and added operational and data-contract complexity.

## Findings That Did Not Survive

- A fixed Stage 1-4 cycle detector is not a sufficient final representation.
- A larger or richer decision head is not automatically better than memory.
- Reliability gating did not solve cross-market transfer.
- The selected global/global reliability filter failed sealed confirmation and
  underperformed raw memory in five of six markets.
- Portfolio exposure control reduced risk but did not create robust alpha.
- A 2+ Sharpe result in one market or one development period is not evidence of
  an international strategy.
- The current final enriched-memory result cannot be interpreted cleanly until
  path quality and inference coverage are repaired.

## Remaining Scientific Questions

- Can bounded path-quality evidence retain the rally-ranking signal without
  outlier domination?
- Does complete inference coverage change India/China/Brazil conclusions?
- Does the memory rank remain positive under per-market drawdown and coverage
  gates?
- Does the top-rank alpha spread persist on an untouched universe or later
  period?
- Can confidence be calibrated so that higher confidence actually means better
  realized outcomes?
- Can raw global-memory ranking retain its limited positive information under a
  new causal, per-market confirmation without a learned reliability filter?

## Final Boundary

At this point the strongest honest research claim is about reproducible market
representation and the conditional usefulness of retrieval-grounded opportunity
ranking, not a confirmed profitable trading system. Any future confirmation
must use a freshly frozen universe/period, repaired causal timestamp handling,
complete inference coverage, and per-market rather than pooled-only gates.

## Final Transport Diagnostic

`configs/temporal_transport_study.yaml` freezes the last untested
representation-level hypothesis. It compares identical global patch
Transformers with and without cross-market, cross-period analogue geometry,
then evaluates both through the C2 minimum-hold policy. See
`docs/TEMPORAL_TRANSPORT_ENCODER_STUDY.md`. This study is diagnostic because the
observed interval has already informed prior research; it cannot restore an
untouched confirmation claim.
