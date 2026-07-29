# Phase 5: Decision Adapter and Historical Reasoning

`[REJECTED] [PARTIAL]`

## Decision Adapter

Phase 5 added a lightweight market-local decision adapter over frozen encoder
latents, targeting a 32-dimensional decision space. The corrected experiment
was `phase5_v2`; `phase5_v1` was retained as a failed control because its
training target used fixed 63-session returns while validation used simulated
A2 exits.

The adapter and memory branches established that retrieval evidence often beat
the direct model head, but performance varied materially by fold and seed. Two
adapter promotion variants reached pooled out-of-fold Sharpe `1.651` and
`1.722`, yet neither passed the complete promotion contract. The latter had
only 2 of 5 retrieval-gate folds, a worst drawdown of `-32.8%`, and did not
improve the underlying A2 path consistently.

## Consensus Policy

The consensus experiments tested median evidence, seed votes, persistent exits,
and upper-consensus decisions without modifying latent coordinates. The
consensus selector was rejected with PBO approximately `0.53`.

## Consensus Exposure: Correct Interpretation

The static C0 median-consensus policy, not an exposure controller, produced the
approximately `2.15` OOF Sharpe with PBO approximately `0.07`. Every tested
controller reduced Sharpe. The useful finding was that the original seed-median
rank policy carried development performance; reactive exposure control did not
improve it. This remained development-only and required external confirmation.

## Opportunity Allocator

The nonlinear opportunity allocator was tested against obvious-signal and
calibrated variants:

- A timestamp-precision audit found lookahead in pre-fix evidence construction.
  The allocator rebuilt its causal evidence after explicit nanosecond
  normalization: 5 folds, 28 seed runs, 0 causal violations.
- In the corrected first allocator run, static B0 Sharpe was `2.123` and the
  deterministic obvious-signal B1 reached `2.149`, reduced drawdown by about
  5.1 percentage points, and was rejected because worst drawdown remained
  `-23.7%` and out-of-year ranking was negative in all five folds.
- Nonlinear learnability won only 2 of the required 4 folds, despite all 5
  folds having a positive nonlinear and obvious-positive count.
- PBO was approximately `0.314` in the first allocator run.
- The final allocator had 0 out-of-year positive folds for every candidate;
  improvement appeared in 4 folds for obvious signal and 5 for calibrated
  obvious signal, but PBO rose to approximately `0.543`.

The allocator was rejected. The core lesson was that the existing signal had
some learnable structure and deterministic risk sizing could reduce drawdown,
but neither nonlinear learning nor calibration produced a temporally durable
ranking of which candidate trades deserved more capital.

## Source Map

- `docs/PHASE5_EXECUTION.md`
- `configs/phase5_decision_alignment_v2.yaml`
- `configs/phase5_consensus_policy.yaml`
- `configs/phase5_consensus_exposure.yaml`
- `configs/phase5_opportunity_allocator.yaml`
- `configs/phase5_final_allocator.yaml`
- `scripts/run_phase5_decision_alignment.py`
- `scripts/run_phase5_consensus_policy.py`
- `scripts/run_phase5_consensus_exposure.py`
- `scripts/run_phase5_opportunity_allocator.py`
- `reports/phase5/phase5_v2/adapter_promotion.json`
- `reports/phase5/consensus_exposure/phase5_consensus_exposure_v1/leaderboard.csv`
- `reports/phase5/opportunity_allocator/phase5_opportunity_allocator_v1/selection_report.md`
