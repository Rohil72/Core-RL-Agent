# Phase 6: International Testbed, Offline RL, and Frozen Memory

`[REJECTED]`

## Purpose

Phase 6 compared regional/global Transformer representations, regional/global
memory, and offline RL policies across US, India, China, Brazil, France, and UK
markets. The durable runner was designed for preemption, checkpointing, source
fingerprints, and resumable jobs on Jarvis Labs.

## Operational Findings

- The A100/A30 hardware was more than sufficient for the pilot; the expensive
  part was the number of full jobs rather than GPU utilization per small job.
- Yahoo Finance data acquisition produced missing ticker/fundamental coverage
  and API failures, so absent earnings and absent tickers had to be audited.
- The durable runner correctly resumed completed and interrupted jobs after
  funding/preemption interruptions.
- An AMP failure in `future_head.2.weight` exposed non-finite gradients; bounded
  recovery/checkpoint behavior was added so long runs did not silently lose all
  progress.

## Offline RL Results

Pilot selection chose IQL and TD3+BC. The full six-market development gate then
rejected every final candidate:

| Candidate | Market wins | Median Sharpe | Median baseline Sharpe | Worst drawdown |
|---|---:|---:|---:|---:|
| IQL global | 4 | 0.057 | 0.051 | -0.345 |
| IQL regional | 3 | -0.077 | 0.118 | -0.292 |
| TD3+BC global | 3 | 0.296 | 0.051 | -0.241 |
| TD3+BC regional | 1 | 0.291 | 0.118 | -0.378 |

The exact selection output recorded `status: rejected`, `selected_candidate:
null`, `confirmation_unlocked: false`, and one overall selection PBO of `0.714`.

This experiment did not reproduce the established C0 policy contract: it
trained independent per-seed allocators on one-day basket rewards and used a
different entry/exit score from the three-seed median-rank system. It rejects
this offline-RL formulation, not every possible RL extension of the consensus
memory architecture. RL was nevertheless dropped from the active path because
it added operational complexity without an observed stable improvement.

## Frozen Memory Topology Sweep

A separate no-RL, no-retraining sweep compared regional/regional,
global/regional, and global/global encoder-memory topologies across six
markets. The best development candidate was global/global with strict 25%
coverage:

| Metric | Value |
|---|---:|
| Pooled Sharpe | 1.503 |
| Mean market return | +17.45% |
| Positive markets | 5/6 |
| Worst drawdown | -21.71% |
| PBO | 0.000 |
| Baseline wins | 3/6 |

It was rejected: profit concentration in China was about 70%, drawdown exceeded
the 20% gate, calibration was weak, and the median market Sharpe was only 0.37.
At full coverage, global/global Sharpe was 0.533, global/regional was 0.472,
and regional/regional was -0.171. Thus regional external memory did not improve
the development result.

Regionalizing both the global encoder's internal memory and external memory
performed worse still: the best dual-regional candidate reached only about
0.695 pooled Sharpe. The evidence favored global representation plus global
memory, but only within the development sweep.

## Sealed Frozen-Memory Confirmation

The selected global/global 25%-coverage candidate was then evaluated in a
locked external confirmation with no architecture, threshold, RL, or training
change. It failed decisively:

| Strategy | Return | Sharpe | Positive markets |
|---|---:|---:|---:|
| Locked reliability-filtered memory | -6.02% | -0.376 | 2/6 |
| Raw memory | +2.37% | 0.218 | 3/6 |
| Equal-weight buy and hold | +9.01% | 0.666 | 3/6 |

The locked candidate's profit concentration was 89.6% in China, and raw memory
beat locked memory in five of six markets. This is the decisive result for the
reliability-filter branch: the development filter did not transfer and actively
destroyed the modest raw-memory advantage.

## Source Map

- `configs/final_research_testbed.yaml`
- `configs/phase6_dual_regional_memory.yaml`
- `configs/phase6_frozen_memory_sweep.yaml`
- `configs/final_memory_confirmation.yaml`
- `scripts/build_final_testbed.py`
- `scripts/run_durable_experiment.py`
- `scripts/train_phase5_offline_policy.py`
- `scripts/evaluate_phase5_offline_policy.py`
- `scripts/run_phase6_frozen_memory_sweep.py`
- `scripts/run_phase6_dual_regional_memory.py`
- `scripts/run_final_memory_confirmation.py`
- `reports/final_testbed/phase6_a30_final_v1/` when restored
- `reports/phase6_frozen_memory/` when restored
- `reports/phase6_dual_regional_memory/` when restored
- External confirmation: `C:\Users\rohil\Downloads\confirmation_summary.json`
