# Phase 4E-I: Memory, Exposure, Reliability, and Rally Entry

`[PARTIAL] [REJECTED] [POTENTIALLY-CONTAMINATED]`

These phases kept Transformer weights frozen and successively tested whether the
remaining bottleneck was retrieval semantics, exposure control, transfer
reliability, entry timing, or neutral-state rejection.

## Phase 4E: Cross-Market Sharpe

Phase 4E targeted pooled out-of-sample Sharpe of 2.0 while freezing transformer
weights and retrieval geometry. The strongest candidate, `A2_base`, reached
pooled Sharpe `1.920` and leave-one-fold-out Sharpe `1.714`; it had positive
mean excess return but a worst drawdown of `-32.6%`. No candidate passed the
predeclared promotion contract.

## Phase 4F: Causal Exposure Controller

The controller consumed frozen A2 signals and combined lagged volatility,
drawdown, and evidence-quality scalars. The evidence controller reached the
best in-sample Sharpe, `1.961`, only `+0.040` above static A2, while its
cross-market OOF Sharpe was `1.852`. Drawdown control reduced worst drawdown to
about `-24.9%` but reduced return and did not create robust excess alpha.

The phase's practical result was negative: trailing risk and memory-evidence
filters removed at least as much upside as downside. This was an early instance
of a pattern later repeated by reliability and abstention layers.

## Phase 4G: Memory Transfer Reliability

Phase 4G fit chronological reliability and error models over memory diagnostics,
then compared 100%, 75%, 50%, and 25% coverage policies. The selected 75%
coverage policy reached Sharpe `1.331`, but OOF Sharpe was only `0.853`, mean
excess return was `-10.3%`, and Brier skill was negative in all five folds.
Top-confidence quartiles underperformed lower-confidence quartiles in every
fold. The `0.69` result belongs to the later dual-regional international-memory
experiment, not this phase.

## Phase 4H: Rally-Start Prototype Memory

Phase 4H created causal successful-rally and failed-setup prototype banks. A
successful prototype required at least +10% future maximum return and reaching
the upside barrier before a -10% drawdown; a failed prototype required the
opposite path. Results retained A2's exit score. The baseline A2 policy
produced Sharpe `1.921`; rally rank fell to `1.640` while reducing worst
drawdown from `-32.6%` to `-24.1%`. The rally gate made the same trades as the
baseline. Rally classification itself was near random (mean ROC-AUC about
`0.503`), so this did not establish a reliable rally-start signal.

## Phase 4I: Neutral-Aware Entry Memory

Phase 4I added a neutral/wait bank so ordinary or too-early states could not be
mistaken for relative success merely because they were closer to success than
failure. Neutral-aware rank raised rally-entry precision from `53.4%` to
`56.1%` and improved worst drawdown from `-32.6%` to `-19.5%`, but reduced
pooled Sharpe from `1.921` to `1.797`; OOF Sharpe was `1.612`. It is a useful
abstention/risk diagnostic, not an alpha engine.

All Phase 4E-I performance reports predate the timestamp-precision repair.
They are retained to explain experimental evolution, but they are not clean
causal promotion evidence.

## Source Map

- `docs/PHASE4E_CROSS_MARKET_SHARPE.md`
- `docs/PHASE4F_CAUSAL_EXPOSURE_CONTROLLER.md`
- `docs/PHASE4G_MEMORY_TRANSFER_RELIABILITY.md`
- `docs/PHASE4H_RALLY_START_MEMORY.md`
- `docs/PHASE4I_NEUTRAL_AWARE_ENTRY_MEMORY.md`
- `configs/phase4e_cross_market_sharpe.yaml`
- `configs/phase4f_exposure_controller.yaml`
- `configs/phase4g_memory_reliability.yaml`
- `configs/phase4h_rally_start_memory.yaml`
- `configs/phase4i_entry_advantage_memory.yaml`
- `reports/phase4e/phase4e_v1/selection_report.md`
- `reports/phase4f/phase4f_v2/selection_report.md`
- `reports/phase4g/phase4g_v1/selection_report.md`
- `reports/phase4h/phase4h_v1/selection_report.md`
- `reports/phase4i/phase4i_v1/selection_report.md`
