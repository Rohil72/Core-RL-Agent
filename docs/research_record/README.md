# Research Record

This directory is the chronological ledger of the market-memory research as
far as the available chat history, repository documents, and supplied report
bundles establish it. It is intentionally separate from the implementation
docs: the implementation docs describe how to run a phase, while these files
record what the phase actually taught us.

## Status Tags

- `[ESTABLISHED]`: supported by a recorded result or a verified artifact.
- `[PARTIAL]`: useful evidence exists, but the result has an important limit.
- `[REJECTED]`: the declared gate rejected the hypothesis or candidate.
- `[DIAGNOSTIC-ONLY]`: inspected evidence cannot support fresh confirmation.
- `[INCOMPLETE-RESULT-RECORD]`: the phase exists in code/docs, but its numeric
  result was not preserved in the material available for this ledger.
- `[NEEDS-REPAIR]`: the result exposed a data or scoring defect that must be
  fixed before its performance can be interpreted as a clean comparison.
- `[POTENTIALLY-CONTAMINATED]`: a later audit found a protocol defect that may
  affect the artifact; it is historical evidence, not promotion evidence,
  until regenerated under the repaired contract.

Combined tags retain each literal status. For example, `[PARTIAL]`
`[POTENTIALLY-CONTAMINATED]` means useful diagnostics remain, but the recorded
performance cannot be treated as clean causal evidence.

## Reading Order

1. [Evidence Boundary](00_evidence_boundary.md)
2. [Phase 1](01_phase1_direction.md)
3. [Phase 2](02_phase2_representation.md)
4. [Phase 3](03_phase3_latent_state_discovery.md)
5. [Phase 4A-C](04_phase4a_c_encoder_and_holdouts.md)
6. [Phase 4D](05_phase4d_retrieval_semantics.md)
7. [Phase 4E-I](06_phase4e_i_memory_policy_ladder.md)
8. [Phase 5](07_phase5_adapter_and_reasoning.md)
9. [Phase 6 International Evaluation](08_phase6_international_rl.md)
10. [Local Rank Ensemble](09_local_rank_ensemble.md)
11. [Final Memory Study](10_final_memory_study.md)
12. [Consolidated Findings](11_consolidated_findings.md)
13. [Q1 Publication Readiness](12_q1_readiness.md)
14. [Temporal-Transport Encoder Study](13_temporal_transport_encoder.md)

The [Phase Report Map](PHASE_REPORT_MAP.md) links each record to the relevant
tracked document, config, script, and report directory.

## Current Bottom Line

The strongest evidence is not a promoted trading strategy. The Transformer
learns reproducible, outcome-separated states, and raw retrieval occasionally
contains useful opportunity information. However, confidence filters, exposure
controllers, rally gates, neutral-state rejection, and regionalization have
not transferred that information reliably across time or markets. The sealed
global/global confirmation rejected the selected reliability-filtered policy;
the final enriched-memory study separately remains invalid pending path-quality
and inference-coverage repairs. The later temporal-transport study improved
causal neighbour quality consistently, but its pooled international trading
metrics require calendar-valuation repair before they can be interpreted.
