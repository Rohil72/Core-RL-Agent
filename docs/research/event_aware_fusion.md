# Event-Aware Financial Modeling Plan

## Executive summary

Current pipeline trains sequence models (LSTM/Conv/Transformer) over 252-day rolling windows using daily technical features and aligned quarterly fundamentals. The active model contract produces a latent state, future-return predictions, and action logits for long-only entry/exit decisions. Recent changes replaced the LSTM default with a patch-based transformer and added updatable slot memory, a fund_report_freshness feature, deterministic data loading, and data-safety asserts.

New goal: center the research and engineering effort around treating earnings and other fundamentals as sparse, high-information events using explicit event tokens and temporal decay, and implement a Temporal Fusion Transformer (TFT)-style model with event-aware attention, variable selection, and memory for regime persistence. The aim is a production-grade long-only entry/exit system and a clean research contribution focused on sparse event-aware fusion.

## Where we are now

- Data & preprocessing
  - Per-ticker precomputed parquet files, loader aligns earnings via merge_asof (backward, no lookahead).
  - Technical features computed daily (SMA windows, momentum, volatility, Minervini template checks).
  - Fundamentals are sparse (quarterly) and were aligned to daily rows; freshness score and report_imminent/just_reported flags added.
- Models & training
  - Patch-based transformer backbone implemented with attention pooling, patch tokens, and persistent updatable memory slots (EMA updates saved in memory_state).
  - Deterministic DataLoader using a seeded torch.Generator.
  - Safety checks preventing feature/target overlap.
  - Tests updated: full pytest suite passes.

## Product requirement (distilled)

- Decisions: long-only entry/exit (multi-day holds), no short intraday actions.
- Signals: dense daily technicals + sparse quarterly fundamentals; fundamentals should be treated as temporally-local, high-value events (freshness decay).
- Strategy: the model should learn which moving-average windows and trend-template rules matter for prediction (not rely on hard-coded thresholds).

## High-level solution

1. Data / Features
   - Flatten ingestion: forward-fill fundamentals onto daily rows within a configurable validity window; also add an explicit event token vector on report dates containing raw report fields (eps_estimate, reported_eps, surprise_pct, total_revenue).
   - Freshness: continuous decay (fund_report_freshness) and binary imminence/just_post flags. Validity window default 21 days (configurable).
   - Candidate MA windows: configurable set W = [20, 50, 130, 150, 200, 252]; compute all and let model decide importance via variable-selection gates.
   - Upweight/oversample days near reports and rows with high template scores to bias learning to signal-rich moments.

2. Model — TFT variant (event-aware)
   - Use core TFT ideas: variable-selection networks per time step, gating mechanisms, static covariate embeddings.
   - Replace LSTM with Transformer encoder blocks for sequence modeling; add event-aware attention heads that prioritize report-event tokens on days they occur.
   - Integrate updatable slot memory (read during forward pass; write via gated/EMA updates after producing latent) to capture market regimes.
   - Support freezing bottom K layers post pretraining; support discriminative learning rates.

3. Heads & objectives
   - Action head: long-only action logits (neutral, enter, hold, exit); enforce min-hold days during evaluation/backtest.
   - Future-return head(s): regression targets as before (MAE/RMSE), with NaN-aware masking.
   - Auxiliary stage head (optional): use weak detector labels for stage classification as a regularizer (careful to avoid detector-reconstruction bias).

4. Training recipe
   - Pretrain (SSL): masked reconstruction + next-window forecasting on 2011–2019.
   - Fine-tune: 2021+ (exclude 2020 by default for engineering; evaluate both including and excluding 2020 for publication robustness).
   - Optimizer: AdamW; scheduler: OneCycleLR or Cosine+Warmup; mixed precision (AMP).
   - Initial LR suggestions: body 1e-4, head 5e-4; batch size 32–128 depending on memory; memory_update_rate start 0.05.

5. Evaluation & publication-quality checks
   - Walk-forward testing with ticker holdouts; per-target metrics and action/confusion metrics.
   - Backtest: realistic slippage & transaction costs; portfolio metrics (CAGR, Sharpe, max drawdown); bootstrap confidence intervals.
   - Ablations: event tokens on/off, memory on/off, learned MA windows vs fixed windows, pretrain vs direct training.

## Phased implementation (high-level)

- Phase A — Data & features (compute per-window MAs, event tokens, freshness, upweighting near reports).
- Phase B — TFT model + memory (variable-selection nets, gating, event-aware attention, memory read/write and freeze API).
- Phase C — Training pipelines (pretrain vs finetune splits; freezing schedule; discriminative LR; AMP & checkpointing of memory_state).
- Phase D — Evaluation/backtest harness and ablation suite; bootstrap CI and statistical tests.
- Phase E — Cloud readiness (Dockerfile, Jarvis Labs launcher, reproducible configs).

## Deliverables

- New model: src/models/temporal_fusion_transformer.py (TFT variant with event tokens + memory).
- Data changes: extended feature computation and event token pipeline in src/data/features.py and loader.
- Training: pretrain -> finetune orchestration in src/trainers/train_cycle_model.py and configs.
- Evaluation: scripts/evaluate_agent.py extended to run backtests and ablations, produce JSON + MD reports.

## Acceptance criteria

- Unit tests pass.
- Local pretrain -> finetune smoke run completes and artifacts saved (model + memory_state + report).
- Backtest shows edge vs baseline with bootstrap CI; ablation identifies contribution of event tokens.

## Questions before starting

1. Confirm pretrain split: 2011-01-01 → 2019-12-31; fine-tune/eval: 2021-01-01 → latest (default exclude 2020). Any changes?
2. Minimum holding policy (enforced at evaluation/backtest): default 5 trading days? specify.
3. Candidate moving-average windows to include: default [20,50,130,150,200,252] — ok?
4. Jarvis Labs: will provide compute credentials when ready; I will prepare Dockerfile and launcher.
5. Remove or archive original LSTM file? current repo keeps it as archive; do you want it deleted from tree?

---

(End of plan document)
