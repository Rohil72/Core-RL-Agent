# Phase 4D: Retrieval Semantics

`[PARTIAL] [POTENTIALLY-CONTAMINATED]`

## Question

With the encoder frozen, does changing the retrieval geometry and outcome
semantics make historical analogues more useful?

## Tested Ideas

- Absolute future return versus universe-relative alpha.
- Lower-confidence-bound evidence scoring.
- Diagonal and low-rank frozen-latent metric adapters.
- Training-only distance calibration and OOD rejection.
- Signal-level seed consensus without averaging incompatible latent coordinates.

The A0/A1/A2/A3/A4 ladder was designed to keep geometry and outcome semantics
separable. A candidate had to pass fold/seed win rates, drawdown, NDCG-style
retrieval gain, paired bootstrap, PBO, and deflated-Sharpe checks.

## Earlier Retrieval Proof

The first market-memory harness, which predates Phase 4D, produced the early
proof that retrieval could beat the direct model head:

| Policy | Return | Sharpe | Profit factor |
|---|---:|---:|---:|
| Retrieval policy | +23.8% | 0.889 | 1.63 |
| Direct model head | -17.4% | not recorded | not recorded |

This was a development result, not a Phase 4D result or a cross-market
confirmation.

## Phase 4D Result

Phase 4D's actual leaderboard is present locally. Relative outcome semantics
helped more than learned retrieval geometry:

| Variant | Mean return | Mean Sharpe | Interpretation |
|---|---:|---:|---|
| A0 legacy identity | 20.1% | 0.950 | Reference |
| A1 universe alpha | 29.2% | 1.430 | Stable semantic gain |
| A2 blended alpha | 36.7% | 1.607 | Best return |
| A3 diagonal metric | 33.8% | 1.661 | Best Sharpe, no retrieval gain |
| A4 low-rank metric | 24.6% | 1.155 | Seed-sensitive |

A2 and A3 both failed promotion because the paired-bootstrap lower bound on
daily improvement remained negative. The learned metric adapters did not
improve NDCG-style analogue quality. The correct lesson is that changing the
outcome definition improved policy behavior; enriching latent geometry did not
establish better historical analogues.

These results predate the timestamp-precision repair and are retained as useful
diagnostics, not as clean causal promotion evidence.

## Source Map

- `docs/PHASE4D_RETRIEVAL_SEMANTICS.md`
- `configs/phase4d_retrieval_semantics.yaml`
- `scripts/run_phase4d_retrieval_semantics.py`
- `reports/phase4d/phase4d_v1/ablation_ladder.csv`
