# Phase 4H: Causal Rally-Start Prototype Memory

## Purpose

Phase 4H tests whether frozen Phase 4C latents distinguish early successful-rally states from comparable failed setups. It does not train or modify the encoder, feature pipeline, oracle, detector, internal transformer memory, or A2 evidence score.

Historical labelled states become external-memory prototypes. A query retrieves successful and failed prototypes separately, then compares their distance-weighted local density. The question is whether a present state resembles prior rally starts more than prior failures.

## Causal Labels

Phase 4H reuses Phase 4C future labels rather than creating a subjective exit rule.

| Prototype | Required realised path | Availability |
| --- | --- | --- |
| Successful rally start | `future_max_return_63 >= +10%` and `event_upside_before_drawdown_126 = 1` | After 126 sessions |
| Failed setup | `future_min_return_63 <= -10%`, `event_drawdown_hit_126 = 1`, and `event_upside_before_drawdown_126 = 0` | After 126 sessions |

`event_upside_before_drawdown_126` records whether the positive barrier was reached before the negative barrier. The 126-session outcome-availability timestamp is mandatory for every stored prototype, preventing future leakage. Phase 4H does not use the 252-session maximum-return target.

Several adjacent states can describe the same realised move. Each bank retains only the earliest state for a ticker and realised event session, using the existing peak/drawdown offset. This prevents a long rally from contributing many overlapping copies of itself.

## Retrieval and Reasoning

For each fold and seed, both banks are built only from mature training rows. The query and banks use the frozen Phase 4E normalization and distance calibration. Retrieval inherits the existing causal constraints:

- 25 nearest prototypes per bank.
- At least 21 sessions between a query and eligible neighbour.
- Same-ticker neighbours excluded.
- At least five eligible successful and five eligible failed neighbours.

Each bank receives the mean Gaussian kernel density of its retrieved distances, using Phase 4E's training-only reference distance as bandwidth. The two densities, adjusted by the training prototype prior, produce `rally_start_probability`, `rally_start_log_odds`, per-bank neighbour counts and distances, density estimates, and an eligibility flag.

The neighbour table preserves the exact historical prototypes used for every query, including ticker, timestamps, role, and distance.

## Predeclared Policy Variants

All variants retain the existing policy configuration. The exit score always remains the frozen Phase 4E A2 `opportunity_score`; only entry ranking or the entry gate changes.

| Candidate | Entry rule | Exit rule |
| --- | --- | --- |
| `R0_a2_base` | Original A2 opportunity score | Original A2 opportunity score |
| `R1_rally_rank` | Eligible states ranked by rally-start probability | Original A2 opportunity score |
| `R2_a2_rally_gate` | A2 ranking only where eligible log-odds are positive | Original A2 opportunity score |

This separation measures entry timing, not an accidental alteration of the exit strategy.

## Evaluation Contract

Validation is the only split used to choose a candidate. The winner must meet the predeclared Sharpe, cross-market, excess-return, drawdown, trade-count, deflated-Sharpe, probability-of-backtest-overfitting, and rally-precision-lift gates in `configs/phase4h_rally_start_memory.yaml`.

Test and holdout remain sealed until every validation gate passes. A failing run writes a confirmation status explaining that confirmation was not executed.

## Run

From the repository root with the existing environment active:

```powershell
python scripts\run_phase4h_rally_start_memory.py --config configs\phase4h_rally_start_memory.yaml --run-id phase4h_v1 --stage all
```

After an interruption, resume the same run directory:

```powershell
python scripts\run_phase4h_rally_start_memory.py --config configs\phase4h_rally_start_memory.yaml --run-id phase4h_v1 --stage all --resume
```

For a one-fold, one-seed source-contract check only:

```powershell
python scripts\run_phase4h_rally_start_memory.py --config configs\phase4h_rally_start_memory.yaml --run-id phase4h_smoke --stage all --smoke
```

The full run produces `source_manifest.json`, prototype audits, signals and neighbour evidence, policy outputs, validation leaderboards, selection reports, and a sealed-confirmation status under `reports/phase4h/<run-id>/`.

## Research References

- [LARA, arXiv:2107.11972](https://arxiv.org/abs/2107.11972)
- [STHAN-SR, AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/16127)
