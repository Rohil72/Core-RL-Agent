# Phase 1: Direction and Initial Analysis

`[INCOMPLETE-RESULT-RECORD]`

## Research Question

The original project explored cycle/stage detection and reinforcement-style
trading. Phase 1 began moving the question toward whether historical market
observations contain reusable structure for future opportunity and risk.

The later research direction explicitly abandoned detector-centric thinking. The
oracle cycle detector became an early experimental tool rather than the final
architecture. The desired system became a Transformer state encoder, searchable
market memory, evidence aggregator, and trading policy.

## What Was Established in the Direction Change

- Trading utility became the primary objective; classification accuracy and
  regression loss became secondary diagnostics.
- Historical analogues, expected upside, downside, holding period, path quality,
  and uncertainty became first-class outputs.
- Evaluation was redesigned around realized return, drawdown, Sharpe, profit
  factor, opportunity precision, and capital efficiency.

## Record Limitation

The available tracked material preserves the direction and the implementation
entry points (`scripts/phase1_analysis.py` and `scripts/orchestrate_phase1.py`),
but not a complete numeric Phase 1 result table. No performance number is
invented here.
