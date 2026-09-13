# Clean memory-system rebuild: implementation and pilot specification

Version: proposal 1, 13 September 2026. Status: **implementation and pilots authorized; production execution not authorized by this document**.

This is a new experimental protocol, not a description of experiments already completed. Numerical defaults below are proposed engineering/scientific choices, not claims that the existing release implements them. Implement the contract, run the restricted local pilots, push the evidence, and obtain a code review. Next prepare VM preflight and guarded launch scripts; production remains blocked until the VM timing/export checks pass and the freeze receipt is approved. Do not overwrite the v1.0.3 manuscript, release, models, caches, or reported results.

Companion files:
- `rebuild_plan/config.proposed.json`: machine-readable choices, counts and unresolved production gates.
- `rebuild_plan/universe_request.csv`: the 108 legacy requested identifiers, including the 103 primary securities.
- `rebuild_plan/feature_contract.csv`: ordered feature definitions for this rebuild.
- `rebuild_plan/acceptance_tests.csv`: required test/evidence inventory.

The prose is authoritative for algorithms and exceptional cases; JSON is its configuration projection. Any disagreement is a validation failure, not permission to choose whichever is convenient. Proposed module paths below are in the experimental code repository, not this manuscript workspace.

## 1. What this run will do

The contribution has two parts: **the memory-system design** and **an empirical test of its incremental economic information**. The system combines historical-window representations, explicit record identity and timing, constrained precedent retrieval, return aggregation, optional neural integration, and a common portfolio policy. Its scientific value is not conditional on every added module improving Sharpe. Conversely, being implemented by the authors does not establish priority over all prior analogue methods: related-work comparisons remain necessary.

This rebuild will:
1. Acquire and freeze one versioned daily-data snapshot with corporate-action and quote-unit metadata.
2. Rebuild features, scaling, labels, fold membership and memory banks from that snapshot.
3. Train both neural backbones from random initialization in every fold and seed; no pretrained Transformer or legacy MLP loading.
4. Give MLP, Transformer, ridge and retrieval the same 42-by-23 annual information window.
5. Fit gates and select fixed mixtures only on their designated development samples.
6. Run a fixed 16-configuration matrix, including a plain k-NN ablation of the memory restrictions.
7. Generate predictions, trades, daily NAV, diagnostics, inference and manuscript tables through one traceable output chain.
8. Export enough evidence to resume locally and verify the analysis without the VM.

**Not included:** open-ended architecture search; intraday/news/options ingestion; reinforcement learning; a new claimed untouched 2025 holdout; changing the universe after viewing results; promising the old 0.972 Sharpe; journal submission during the compute window.

Before production, provide `architecture_lineage.csv` mapping each claimed hierarchical ingestion/memory component to its implemented module, mathematical operation, input availability and ablation. Identify which earlier components are outside this study. Do not silently remove an author-claimed component or relabel a metadata hierarchy as a learned hierarchical model. Unresolved lineage blocks the final architecture description and freeze.

## 2. Freeze, dates and evidence status

### 2.1 Historical evaluation is not a new untouched test

The supplied project audit reports earlier P0–P6 results through 30 December 2025. It also reports 2024-query labels extending into April 2025. Therefore, 2025 cannot be certified as untouched at project level merely because Study 2 did not trade in that year. Whether finite 2025-maturing labels entered earlier forecast summaries must be audited, rather than inferred from date maxima alone.

The new contribution is a consistently implemented, temporally admissible walk-forward evaluation. A short Methods statement records prior outcome exposure; a research paper need not narrate the development history. No file or report may call this a prospective experiment or an untouched confirmation sample. Do not treat rerunning as erasing prior information.

### 2.2 Six annual folds

All dates below are inclusive local-session dates. A calendar cutoff means the last relevant exchange close on or before that date. Actual availability timestamps control admission.

| Evaluation year | Backbone/scaler history starts | Training label cutoff | Checkpoint-validation queries | Gate/mixture-development queries |
|---|---|---|---|---|
| 2020 | 2013-01-01 | 2017-12-31 | 2018-01-01 to 2018-12-31 | 2019-01-01 to 2019-12-31 |
| 2021 | 2013-01-01 | 2018-12-31 | 2019-01-01 to 2019-12-31 | 2020-01-01 to 2020-12-31 |
| 2022 | 2013-01-01 | 2019-12-31 | 2020-01-01 to 2020-12-31 | 2021-01-01 to 2021-12-31 |
| 2023 | 2013-01-01 | 2020-12-31 | 2021-01-01 to 2021-12-31 | 2022-01-01 to 2022-12-31 |
| 2024 | 2013-01-01 | 2021-12-31 | 2022-01-01 to 2022-12-31 | 2023-01-01 to 2023-12-31 |
| 2025 | 2013-01-01 | 2022-12-31 | 2023-01-01 to 2023-12-31 | 2024-01-01 to 2024-12-31 |

For evaluation year Y:
- Training origins start in 2013; admit only targets whose 63-session outcome is available by 31 December Y−3.
- Validation queries occur in Y−2; retain only labels available by the end of Y−2. Late-year queries with later labels do not select checkpoints.
- Development forecast/gate queries occur in Y−1; retain only labels available by the end of Y−1.
- Development portfolio selection uses only NAV observations and fills within Y−1, with the explicit year-end liquidation rule below. It does not require every forecast to have a mature training label.
- No fitting on validation plus development after checkpoint selection. Such a refit would change the frozen predictor supplying gate inputs and require a different protocol.
- Training records, validation records, development records and evaluation queries have disjoint origin-date roles within each fold. Overlapping historical input context is allowed; using unavailable target outcomes is not.
- The bank is frozen at the training cutoff for both development and evaluation of that fold. It refreshes annually, not during the evaluation year.
- All markets use timestamped availability in UTC when information crosses markets. Local dates alone must not authorize use of another market's not-yet-observed outcome.

Every fold manifest contains counts, origin bounds, latest input time, latest target availability and excluded-by-reason counts per market. Impossible timestamps or empty market partitions fail the stage.

## 3. Data we will retrieve

### 3.1 First acquisition, fixed scope

Start with the legacy 108 requested identifiers in the companion CSV: **103 primary securities across US, India, China, Brazil, France and UK**, plus five historically excluded identifiers for coverage auditing only. Do not automatically add those five or drop a failing primary security. Historical exclusion reasons in the CSV are inherited statements, not a new verification of corporate events.

Request daily records from **1 January 2010 inclusive through 30 April 2026 inclusive**; use **1 May 2026 as the exclusive API end**. Roles:
- 2010–2012: feature warm-up only, not training origins.
- 2013–2025: fold-specific history, development and evaluation.
- 1 January–30 April 2026: label-completion tail for 2025 predictions only. No 2026 portfolios, regime selection, fitting or new query forecasts.

Verify that each scored 2025 origin has its actual 63rd subsequent exchange session. The nominal tail end is not a guarantee of coverage. Missing maturity is reported; extend the tail only under a logged coverage amendment. Do not invent forward returns or use zero for unknown labels.

### 3.2 Candidate provider and explicit retrieval settings

Implement a per-security Yahoo Finance/yfinance adapter for acquisition and pilots. Pin its installed version and transitive environment in the retrieval manifest before downloading. Do not depend on current default adjustment behavior. Proposed request settings are daily interval, `auto_adjust=False`, `back_adjust=False`, `repair=False`, `actions=True`, `keepna=True`, and `prepost=False`. Use one concurrent request initially; four total attempts, with 2, 8 and 32 seconds between attempts. Record failures and terminate the acquisition stage if required primary-security objects remain missing. The five audit-only excluded symbols may legitimately return no data; preserve a signed-off no-data record rather than treating their absence as a missing primary series.

Retain exactly returned OHLC, adjusted close if supplied, volume, dividends, splits, timezone/exchange/currency metadata, request parameters, retrieval UTC and content hashes. Retain the original response-derived tables and separately version normalized tables. Never silently repair the preserved snapshot. Record renamed, redirected and missing tickers without replacing them with another security.

**Critical distinction:** requesting `auto_adjust=False` does not establish that historical quotes are nominal as-traded prices, that volume has a particular split basis, or that the universe is point-in-time. The production adapter must prove its price/action convention with fixtures and selected source checks. Otherwise it is a pilot-data adapter only. An independently documented vendor can be substituted before freeze without changing the downstream schemas.

Underlying data access and redistribution rights must be checked before publishing a raw-data bundle. A software license is not permission to redistribute vendor prices. If raw redistribution is unavailable, preserve a private checksummed snapshot and publish the acquisition recipe plus permitted derived artifacts.

### 3.3 Required data objects

| Object | Required fields / contract |
|---|---|
| `security_master.parquet` | Stable project security ID, vendor symbol, market, venue/MIC, timezone, account currency, vendor quote unit, quote-to-account multiplier, verified symbol aliases, primary-universe flag, metadata evidence. Unknown quote units are not defaulted to 1. |
| `sessions.parquet` | Venue, local session date, UTC open/close, session status. Use a verified exchange schedule with holidays and exceptional closures. Validate against observations. Never substitute weekdays or compress gaps into shorter horizons. |
| `provider_bars.parquet` | Security ID, session, vendor OHLC/adjusted close/volume, source hash and retrieval batch. Preserve missing values. |
| `corporate_actions.parquet` | Security, effective session/time, action type, split new-shares/old-share ratio, dividend amount and currency, dividend per-share basis, optional payment date, evidence and verification state. Unsupported restructurings are explicit failures needing treatment. |
| `canonical_bars.parquet` | Verified as-traded OHLC in account units, shares-volume convention, tradability, bar-validity flags, action-adjustment provenance. Do not double-apply a split already embedded in vendor prices. |
| `total_return_bars.parquet` | Causally constructed feature/label price series and action-adjusted volume, with transformation version. Not fill prices. |
| `data_audit.json` | Coverage, gaps, invalid/non-positive prices, actions, redirects, duplicates, quote-unit checks and unresolved events per security. |

Required metadata is not all established by the old ticker manifest. The companion CSV deliberately leaves those fields unverified. Populate and validate them; do not fabricate venue, issuer or pence/pound conventions.

### 3.4 Canonical price/action convention

Production mode is `as_traded_with_actions`. An explicit reviewed amendment is required to use research-price execution instead; there is no automatic fallback.

Let S be new shares per old share effective before today's open, or 1 without a split. Let D be today's cash distribution normalized to **one pre-action share**, in account currency. For consecutive valid sessions, set total-return gross growth to `(S * close_today + D) / close_previous`. Starting at 100 for each uninterrupted valid segment, multiply the previous total-return close by this growth. Map open/high/low with the same positive affine transformation `previous_TR_close * (S * raw_price + D) / previous_raw_close`. For volume features, divide as-traded volume by cumulative split factor within the segment. Require verified dividend/split basis when both occur on a session. This is the new feature/target convention, not a claim of numerical equivalence to the old adjusted-price cache.

No future action may change an earlier feature row. Test this by appending future splits/dividends and checking earlier outputs byte-for-byte. Unknown complex actions, missing action amounts or ambiguous adjustment basis block the affected production data path. Pilot synthetic data can still exercise all downstream modules.

### 3.5 Missing data and universe limitations

- Reject non-positive/non-finite OHLC and inconsistent high/low bounds; reject negative volume. Zero volume is retained as zero and makes a bar non-tradable for this protocol.
- Exact duplicate records may collapse with an audit count; conflicting duplicates fail.
- Missing sessions remain missing. No backward fill. No zero-price sentinel. Do not replace missing volume with 1.
- Restart feature warm-up after a missing/invalid scheduled bar. This conservative rule is intentional and its exclusions must be counted.
- A held security may be causally valued at the last valid close, adjusted for verified actions, but that is not a tradable quote. More than five consecutive stale held sessions triggers a data-treatment review before certifying a production result. Preserve the partial ledger; do not force an invented liquidation.
- Fixed membership in the 103-security list remains survivorship-conditioned. A point-in-time-universe study requires historical membership and delisting coverage not provided by this plan. This limitation cannot be removed by retraining.

### 3.6 What is not in the first download

No intraday ticks, option chains, news, embeddings, fundamentals or FX are needed for the primary local-currency portfolios. No US-dollar pooled portfolio is claimed. Factor-adjusted alpha is a separate extension: it requires suitable market/region factors, frequency, risk-free series and currency alignment. Do not apply US Fama–French factors indiscriminately to six local-currency markets. Implement an optional factor-data interface, but mark that analysis `NOT_RUN` until a reviewed factor manifest exists. Do not advertise the primary Sharpe contrasts as factor alpha.

## 4. Feature, window and target contract

`feature_contract.csv` defines all 23 columns in order. Retain the known duplicate peak-distance geometry for this bounded experiment; do not remove it after inspecting results. Rename the misleading 21-day drawdown field to `drawdown_252` in the new schema, retaining its legacy alias in the dictionary.

All rolling windows require their full length. Feature standard deviations use `ddof=1`. Denominator guard is 1e-9. EMA uses `adjust=False` with its first valid close as the initial value. RSI uses simple 14-session gain/loss means, normalized to [0,1]; a completely flat window returns 0.5. Volume change is zero when both consecutive volumes are zero, invalid when the previous volume is zero and today's is positive. Invalid feature rows are masked, never filled with a typical value.

Require at least 252 contiguous valid bar sessions before accepting a feature row, and enough additional history for its formula, including return lags. A query additionally needs 252 consecutive valid feature rows. Report the resulting warm-up loss rather than assume every requested security contributes every day.

For each fold and market, estimate mean and population standard deviation from unique valid feature rows dated from 2013-01-01 through the training cutoff only. These unsupervised scaler rows need not have a mature label, but may not cross the training cutoff. Use `max(std,1e-4)`, standardize then clip to [-5,5]. Persist float64 moments; transformed rows are float32.

A decision after local close t uses feature rows t−252 through t−1, excluding row t. Pool consecutive groups of six rows to produce `(42,23)`. Transformer receives that matrix; MLP and ridge receive its row-major 966-vector. Retrieval uses the same flattened vector. Training, validation, bank construction and query extraction call the same function. Reject malformed dimensions and insufficient windows; never implicitly unsqueeze a vector into a Transformer sequence.

Target: `TR_close[t+63] / TR_close[t] - 1`, decimal units, using the 63rd subsequent exchange session. This is a total-return forecasting target, not realized next-open trading P&L. Store `label_value`, `label_available_utc`, `origin`, `horizon` and `label_valid` separately from inputs. Memory additionally requires the 126-session availability date by its bank cutoff; no 126-session regression target is trained. A target is valid only when every scheduled bar from its origin through its 63rd subsequent session belongs to the same uninterrupted valid total-return segment. For memory admission, require an observed valid bar through the 126th session in that same segment, not merely a calendar date in a future schedule. Evaluation labels are joined only after predictions have been sealed.

## 5. Fresh neural training and ridge

One global model is fitted across the six markets for each fold, architecture and seed. Seeds are 7, 17 and 37: **6 folds × 2 architectures × 3 seeds = 36 selected backbone fits**, not 216. All epochs and failed/restarted attempts are separately logged.

### Architectures

- MLP: Linear(966,64), LayerNorm(64), GELU, Linear(64,128), LayerNorm(128), scalar Linear head. No second GELU and no dropout. This replaces the old 23-input MLP for matched temporal coverage.
- Transformer: Linear(23,64), sinusoidal positions with base 10000, two post-norm encoder layers with four heads, feedforward width 128, GELU, dropout 0.1; learned scalar attention pooling over 42 tokens; Linear(64,128), LayerNorm(128), scalar head. LayerNorm epsilon 1e-5. Assert input `(B,42,23)`.
- Explicit initialization: all Linear weights Xavier-uniform and biases zero; LayerNorm weight 1/bias 0; Transformer attention projection weights Xavier-uniform and biases zero. Initialize each cloned encoder layer separately under the seeded generator. No checkpoint import before a fresh fit.

### Fixed fitting rule

AdamW: learning rate 0.001, weight decay 0.0001, betas(0.9,0.999), epsilon 1e-8. Effective batch 512, shuffle once per epoch without replacement, retain the final partial batch. Gradient norm clip 1.0. No scheduler. Maximum 50 epochs, minimum 5 epochs, patience 5 validations without an improvement exceeding 1e-6 absolute validation loss. Validate after every epoch. Update the saved best checkpoint only on an improvement exceeding that threshold; earliest checkpoint wins ties. Stop after minimum epochs once patience is exhausted. Record the complete trajectory and selected epoch.

Training objective is equal-market mean squared error: with N total rows and N_m rows in market m, row weight is `N/(6*N_m)`; mean the weighted losses over the epoch. Gradient accumulation scales microbatch losses by their actual row count divided by the actual macro-batch count. Validation computes each market's ordinary MSE then averages six markets. No label standardization; forecasts and targets remain decimal returns.

FP32 training is the initial canonical mode, AMP disabled, TF32 disabled. Set all Python/NumPy/Torch seeds; request deterministic algorithms and fail on unsupported operations rather than silently relaxing them. Record device/library versions. Identical seeds do not promise identical learned weights across platforms.

Initial microbatch: 64 on laptop, 256 on A30; effective batch remains512. Pilot changes to microbatch/workers are operational settings recorded before freeze. Dropout means changing batching can change random trajectories, so hardware equivalence means contractual equivalence, not identical trained parameters. No full training dataset copied to GPU; batch from shared arrays. Start with two loader workers on laptop, four on VM; profile before increasing.

For ridge, use the same 966 inputs and decimal target. Fit one deterministic model per fold with an unpenalized intercept and objective equal-market MSE plus `0.001 * squared_L2_norm(coefficients)`. Solve weighted centered normal equations in float64; do not invoke another hyperparameter search.

## 6. Memory bank, retrieval and mixtures

### 6.1 Shared bank and ordering

For each fold, retain valid origins from 2013 onward whose input window exists and whose 126-session outcome availability is at or before the training cutoff. The stored value is the mature 63-session target. Store stable IDs, security identity, venue/session ordinal, origin/availability times, pooled vector, value and source hashes. Use one physical bank per fold, shared across all model seeds and policies.

Similarity distance is squared Euclidean distance on stored float32 vectors, evaluated with a float64 reference implementation. Order by distance then stable record ID; no unstable top-k tie handling. Exclude the query's exact canonical security, including verified historical aliases. Do not claim issuer-wide exclusion without an issuer mapping.

Primary `MEM_SIM` accepts the first 25 records with at most 3 per security and pairwise separation of at least 21 origin sessions for records of that security. Start with a candidate buffer 250, double until enough accepted or the eligible pool is exhausted. Provisional buffer exhaustion is not inadequate bank coverage. If fewer than 25 survive the complete scan, emit the global bank mean with an explicit fallback flag; never silently average an undersized set.

`KNN_PLAIN` uses the same bank, representation, value, k=25 and query-security exclusion, but removes the per-security cap and 21-session spacing. It is an ablation of those restrictions, not a second unrelated predictor.

`MEM_RANDOM` shuffles the query-eligible records without replacement and applies the primary cap/spacing rules. Use PCG64 with an integer seed derived from SHA-256 of bank hash, fold, query ID and master seed 1001/1002/1003. Specify UTF-8 canonical JSON encoding and use the first16 digest bytes as a little-endian integer. Expanding a buffer continues the same permutation; it must not redraw earlier priorities.

`HIST_PRIOR` is the unconditional mean of all valid63-session values in that fold's bank. It is not a ticker-conditioned mean. Its constant cross-sectional forecast has undefined rank correlation, not IC zero.

The canonical reference distance subtracts the stored float32 vectors promoted to float64, squares elementwise and sums across all 966 features with NumPy float64 reduction. Final candidate ordering always uses this direct reference calculation and stable record-ID ties.

For acceleration, use chunked float64 matrix products to propose candidates via `||q||^2 + ||bank||^2 - 2*q.dot(bank)`. On A30 run those products on CUDA; on laptop allow CPU BLAS or CUDA, choosing by the timing pilot before freeze. Start with128 queries and16384 bank records per distance chunk. Reuse bank norms. Keep only candidate prefixes and boundary information, not a full query-bank distance archive.

For each provisional buffer of size B, find its approximate Bth distance and include **all** records whose approximate distance is at most that boundary plus2e-6, including tied records from other chunks. Recompute those distances using the CPU direct reference, sort, then apply restrictions to the first B reference-ordered records. Expand B when necessary. The error budget is1e-6 absolute per matrix-product distance for finite clipped966-vectors; each included and audited distance must satisfy it. Test that bound against the full reference on every real pilot query and adversarial cases. A violation disables the accelerated backend and requires review; never silently widen the budget after seeing selected outcomes. Values below−1e-6 before refinement are failures; tiny negatives can be clamped to0 only for candidate proposal, never retained as the final reference distance. This bounded-input float64 path does not permit FP32/TF32 approximate ranking. Compare the final neighbor IDs exactly against the reference in all pilot cases. Save selected IDs/distances/weights, not a quadratic distance archive. Approximate nearest-neighbor indexing is out of scope.

### 6.2 Gate

For each fold, backbone and seed, compute development predictions from its selected frozen checkpoint and frozen bank. Inputs are exactly `(|base|, |base−memory|, |memory|)`, no constant channel and no realized error/return. Standardize using development population moments with standard-deviation floor 1e-4. Gate `g=sigmoid(a'z+b)`; forecast `(1−g)*base+g*memory`.

Initialize a=0, b=log(1/3). Fit 50 full-batch Adam steps, learning rate 0.05, betas(0.9,0.999), epsilon 1e-8, weight decay 0.001. Objective is equal-market development MSE. No evaluation-year early stopping. Save inputs/normalizer/weights/loss path. This yields 36 gate fits. This is a proposed bounded fitting rule, not the old gate's training history.

### 6.3 Two fixed-mixture selectors

Grid: memory coefficient λ in {0,0.10,0.25,0.50,1}. Fit no additional backbone for a mixture.
- `MIX_MSE`: minimize development MSE, averaging first across seeds within each market and then across markets.
- `MIX_SR`: maximize development net portfolio Sharpe under exactly the primary execution/cost rules, averaging in the same seed-then-market order.

Select one λ per backbone/fold for each objective. Compare full-precision values. Values within1e-12 of the optimum are tied; choose the smallest λ. Export all grid rows and selected IDs. A selected endpoint identical to another arm is an alias, not an independent experiment. For development portfolio selection, use a fresh 100000-unit account for that development year, no incoming positions, and the terminal rule in Section8.

k10 and k50 sensitivity is restricted to development diagnostics in this first compute budget. k=25 stays the primary choice. No sensitivity-driven retrospective selection of the winning evaluation arm is allowed.

## 7. Complete primary configuration matrix

| Configuration | Realizations per market | Role |
|---|---:|---|
| MEM_SIM |1| Proposed constrained pooled-window memory |
| KNN_PLAIN |1| Same k-NN without cap/spacing |
| MEM_RANDOM |3| Random-priority precedent control |
| HIST_PRIOR |1| Unconditional bank mean |
| RIDGE_ANNUAL |1| Linear matched-information predictor |
| MLP_BASE |3| Fresh matched-window MLP |
| TRANS_BASE |3| Fresh Transformer |
| MLP_MIX_MSE |3| Forecast-selected constant mixture |
| TRANS_MIX_MSE |3| Forecast-selected constant mixture |
| MLP_MIX_SR |3| Portfolio-selected constant mixture |
| TRANS_MIX_SR |3| Portfolio-selected constant mixture |
| MLP_GATE |3| Adaptive MLP-memory integration |
| TRANS_GATE |3| Adaptive Transformer-memory integration |
| MOMENTUM_21 |1| Plain21-session momentum ranking |
| VOL_MOMENTUM_21 |1| Volatility-normalized momentum |
| PASSIVE_EQUAL_WEIGHT |1| Initial equal-weight buy-and-hold |

There are 7 deterministic realizations, 3 random-memory realizations and 24 neural-derived realizations: **34 per market,204 continuous market-policy paths,1224 logical market-year cells**. The plain k-NN control is the explicit addition to the previously discussed15-arm scope. Counts are logical before alias elimination. Development grids, pilots and sensitivity runs are additional work and must appear separately in the runtime estimate.

A positive historical-prior score divided by volatility already ranks inverse volatility. Do not add a nominally separate identical policy and claim independent confirmation. A non-positive prior produces no new entries under the positive-score rule below.

## 8. Execution contract

This is a transparent daily simulation, not institutional execution certification. Board-lot/tax/impact realism is not inferred from penny reconciliation.

### 8.1 Accounts, calendars and ranking

Run one continuous local-currency account per market-policy realization from its first 2020 session through its final 2025 session; initial capital 100000. Update model/bank parameters at each year's start, without resetting positions, holding ages or economic state. Orders already queued at the preceding close retain their outgoing-model instructions; subsequent close decisions use the incoming fold. Seed IDs remain paired across annual refreshes. Report annual panels from these continuous paths, not independently reset annual portfolios.

The first entry signal is generated at the close of the last pre-start session using the incoming fold's artifacts, solely for execution on the first evaluation open. Tag that query `entry_warmup`, exclude it from evaluation forecast metrics. This applies also to each standalone development portfolio. Training is a simulation of information availability, not a claim the experiment actually ran before those dates.

At close t, active policies rank valid, not-currently-held securities. All active arms share the same query-validity mask and availability of21-session daily volatility and14-session ATR. Prediction policies use `forecast/(vol21+1e-4)`; plain momentum uses the 21-session total-return change; normalized momentum divides that change by the same denominator. Volatility and ATR may include today's completed bar, although prediction features stop at t−1. Only strictly positive finite scores are eligible. Ties use ascending security ID.

Maximum 3 occupied positions. An exit queued for tomorrow still occupies a slot today; do not anticipate freed slots. Select at most the currently empty slots, with no duplicate/pyramiding entries. Entry instructions expire if the next scheduled open is non-tradable. Exit instructions remain pending until a valid tradable open.

### 8.2 Fills and funding

Base commission f=0.001 per side; base slippage s=0.0005 per side. These are explicit new assumptions. For an entry planned at close t, per-slot budget is E_t/3. At the next open, buy fill is `open*(1+s)`; quantity is the minimum of `floor(0.95*budget/[fill*(1+f)])` and the available-cash equivalent. Quantities below1 do not trade. Process pending exits first, then entries in the prior decision's rank order. No leverage/negative cash; fees are debited explicitly. Keep prices, quantities, cash and NAV in float64 without rounding the ledger to cents; round only presentation tables. Independently reconstruct production cash and NAV and require agreement within 1e-6 account units per session; the small hand-calculated fixtures use the stricter tolerance in the acceptance inventory. Sell fill is `open*(1−s)` with commission on gross sale notional.

Entry lot size 1 is a research convention, not a claim about every venue's board-lot rules. Action-created fractional quantities are allowed and sold in full; ordinary new entries remain integer shares. Normalize quote subunits to account currency before all sizing, fees and valuation.

### 8.3 Actions, stops and missing quotes

At an action-effective session before open, for existing holdings credit `old_quantity*D` to cash, multiply quantity by S, and divide per-share cost basis by S. Cash dividend credit at ex-date is an explicit approximation to payment-date financing. A verified payment-date implementation is a separate amendment, not a silent switch.

Adjust the stored peak into today's ex-action share-price basis as `max(1e-9,(old_peak−D)/S)` before comparing today's price. On a new entry, initialize peak to the raw execution-session open, then update at closes. This is the specified dividend-adjusted Chandelier-style rule, not an inference from the legacy engine.

At every close, peak=max(previous adjusted peak,current valid close). Entry session counts as holding age 1; age increments on every scheduled local session, including a halt. Queue an exit for the next open when `close/peak−1 < −max(0.10,2.5*ATR_ratio14)` or age≥63. A missing close cannot trigger a fabricated price stop; the age rule still queues an exit. Apply the stale-valuation review rule in Section3.5.

No annual liquidation during 2020–2024. For the final 2025 year and for each development portfolio, stop issuing new entry instructions during the last five scheduled sessions; liquidate remaining holdings at the final valid session close with the same slippage and fee. Record this as a synthetic terminal fill. A missing final quote blocks certification until treatment is reviewed; do not use a future price.

### 8.4 Passive reference and cost stress

Passive equal weight purchases at the first portfolio open, assigning equal initial budgets to primary securities with valid positive tradable opens at that date. Use integer shares and the same fee/slippage; residual cash remains cash. Apply the same corporate-action and missing-valuation rules. No periodic rebalancing or later IPO entry. Initial eligibility and counts are disclosed separately from the active feature-validity mask. Terminal sales use the same terminal rule.

Mandatory cost stress changes slippage to 0 and0.0015 with commission fixed 0.001, for MEM_SIM, HIST_PRIOR, MLP_GATE and TRANS_GATE only. Re-execute their full ledgers with frozen signals; do not subtract a cost estimate from an already selected path. Different cash balances may alter quantities. These are 48 additional continuous market-policy paths per stress setting, 96 in total, not additional model fits. No holding-horizon or leverage search in this first run.

## 9. Metrics, inference and robustness

Every metric must be recomputed from the new release; do not carry old MSE, exposure, gate bins or turnover into new tables.

### 9.1 Portfolio and forecast estimands

Use r_t=E_t/E_(t−1)−1, including initial costs against initial capital. Common annualization252 for all markets is a convention; actual rows follow native calendars. Annualized return=`exp(252*mean(log1p(r)))−1`; volatility=`sqrt(252)*population_std(r)`; Sharpe=`sqrt(252)*mean(r)/population_std(r)`, with zero assigned if standard deviation ≤1e-8. State zero risk-free subtraction, not factor alpha. E≤0 or r≤−1 fails the primary unlevered accounting run rather than clipping log returns.

Maximum drawdown includes initial capital in the running peak. Win rate is the fraction of sessions r>0. Annualized turnover is `(252/N)*sum(abs(executed_gross_notional))/(2*mean(daily_E))`, including initial and terminal fills but excluding mechanical splits/dividends. Exposure is the mean of risky-position market value/E. Save these formulas in analysis metadata.

For each arm, compute each realization's statistic within each market, average realizations within market, then average six markets. Headline Sharpe is computed from the continuous2020–2025 returns for each realization, not an average of annual Sharpes or the Sharpe of pooled market returns. Year/market tables are complementary.

Primary forecast summaries use common valid labeled-query masks. MSE is averaged by market then realization consistently. Daily within-market Spearman IC requires at least 5 observations and nonconstant forecasts and outcomes. Undefined IC is missing with a reason, never set to0; report defined-date counts. Average defined daily ICs within market, then markets and realizations with explicit coverage. Label-based metrics do not grant those labels to the trading engine.

Save gate weights, base/memory forecasts and realized errors keyed by fold, seed, market, security and origin. New gate diagnostics must identify this population; do not reuse the old 25,753-bin table. Prediction-versus-investment rankings are results, not predetermined conclusions.

### 9.2 Eight primary Sharpe contrasts

P1 MEM_SIM−MEM_RANDOM; P2 MEM_SIM−HIST_PRIOR; P3 MEM_SIM−KNN_PLAIN; P4 MEM_SIM−RIDGE_ANNUAL; P5 MEM_SIM−MLP_BASE; P6 MEM_SIM−TRANS_BASE; P7 MLP_GATE−MLP_MIX_SR; P8 TRANS_GATE−TRANS_MIX_SR.

These answer selection, local conditioning, memory restrictions, modern/simpler predictors and adaptive versus portfolio-selected fixed integration. Forecast-MSE comparisons and MIX_MSE comparisons are reported descriptively; do not opportunistically add uncorrected significance claims.

Use 10000 synchronized moving calendar-week bootstrap draws, primary block length4 weeks; block sensitivities2 and8 weeks. Sample separately within each of the six calendar-year strata, with non-wrapping consecutive-week blocks chosen uniformly from all valid starts, concatenating and truncating to the stratum's original number of weeks. A week is its Monday-date bucket; partial boundary weeks retain their actual observations. Apply the same sampled week IDs to every market, policy and seed. Retain native-market holidays as absent observations, not invented zero-return sessions. Concatenate the sampled strata for the continuous-period statistic. This conditions on the observed year mix and is not a general future-regime coverage guarantee.

Initialize `PCG64(SeedSequence([42, block_length]))` separately for each block length. Save the sampled week index arrays. For contrast estimate theta and draw estimates theta_b, interval=`theta ± quantile(abs(theta_b−theta),0.95,method='linear')`; two-sided centered p=`(1+count(abs(theta_b−theta)>=abs(theta)))/(B+1)`. Apply Holm to the eight primary p-values only. Intervals are marginal, not simultaneous Holm-adjusted intervals.

Compute sufficient statistics in float64: counts, sum returns, sum squared returns and sum log gross returns. Clip negative variance to0 only when its magnitude ≤1e-12; larger negatives fail. A draw with zero observations for any required market fails rather than being silently replaced. Statistical replay tolerance1e-10 applies to the same archived predictions/returns and defined numeric environment, not to independently retrained GPU weights.

### 9.3 Predefined diagnostics

Produce six market tables, six yearly tables and per-market leave-one-market-out summaries of the other five markets. Report first/second-half panels without treating them as six additional independent tests. Build the diagnostic market index by compounding the daily arithmetic mean of primary-security total returns with valid consecutive bars, requiring at least 5 securities; otherwise its daily return is missing. It is a diagnostic index, not an executed portfolio. Describe volatility conditions using this index: high volatility means prior-close21-session volatility exceeds that fold's development median; low is the complement. Bull/bear state is prior-close index above/below its trailing200-session mean. Fit thresholds only on development history. Use full 21-return and 200-close histories; missing index history produces an undefined state, not a filled label. Apply state labels causally; report sample sizes and descriptive metrics without searching thresholds.

Do not add size/liquidity subgroup claims without corresponding historical metadata. Do not use these diagnostics to remove a difficult market or select the headline arm after the fact.

## 10. Code architecture and artifact interfaces

Implement a new `memory_study_v2/` namespace and `tests/memory_study_v2/`. Read legacy modules for lineage only; production must not import legacy checkpoint-loading functions or machine-specific paths. All paths are CLI/config-relative roots. Environment, data and output paths must never be embedded in scientific modules.

| New module | Responsibility and principal output |
|---|---|
| `contracts.py` | Validate config/schema, finite enums, required fields, production gates; reject unknown keys. |
| `acquisition.py` | Provider requests, preserved snapshots, retries, raw manifests. |
| `canonical_data.py` | Calendar alignment, quote units, actions, canonical and total-return bars. |
| `features.py` / `representations.py` | Ordered features, frozen scalers, one shared annual window extractor. |
| `folds.py` / `labels.py` | Availability-enforced roles and separately stored labels. |
| `artifacts.py` | Hashes, lineage DAG, atomic states, compatibility and resume. |
| `backbones.py` / `train.py` | Strict model shapes, fresh fits, validation selection and checkpoints. |
| `memory.py` / `retrieval.py` | Bank admission, exact/reference retrieval, controls and saved neighbor evidence. |
| `integration.py` | Gate fits and the two development selectors. |
| `predict.py` | Sealed, label-free policy predictions and query masks. |
| `execution.py` | Event ordering, account state, actions, fills, stops, valuation. |
| `metrics.py` / `inference.py` | Single metric implementation for selector, evaluation and replay. |
| `reporting.py` / `release.py` | Tables and private/public artifact inventories, checksums, replay. |
| `pilot.py` | Synthetic validation, bounded real-data pilots, stage timings and budget projection. |

Stage dependency chain: raw snapshot → canonical data → features/fold/scalers → train/bank → validation-selected predictors → development integration/selection → sealed evaluation predictions → portfolios → metrics/inference → reports/release. Explicitly allow the selector to call the same execution/metric functions without granting evaluation data access.

Stable query key=(fold,security_id,origin UTC,target version). Prediction key additionally includes configuration and realization. Prices/actions never join labels inside execution. Make accidental evaluation-label access fail through the data-access API; directory names alone are not an access control.

Each artifact records schema version, scientific config hash, operational config hash, source commit, dirty-diff hash, environment digest, parent hashes, row counts, date/availability bounds and completion state. Canonical JSON uses sorted keys, compact separators, UTF-8, and rejects NaN/Infinity. File digest is SHA-256. Parquet byte identity is not cross-library guaranteed: record file hashes plus sorted semantic-content hashes under a defined serialization.

Write temporary files, fsync/close, then atomically rename and mark COMPLETE. Existing paths are not proof of completion. On resume require matching parents/config/environment policy; never silently accept a checkpoint because its representation string matches. Save best and last checkpoint, optimizer, RNG and sampler state at completed optimizer-step boundaries. Handle preemption with a valid last checkpoint or restart the incomplete epoch under a logged policy; no duplicated completed updates.

Alias outputs identify their canonical artifact explicitly. Do not count deterministic duplicate endpoints or repeat seeds as new independent evidence. Public release includes forecasts if permitted, full-precision metrics, ledgers/returns as permitted, neighbor manifests, configuration and analysis draws. Private reproducibility archive additionally keeps canonical inputs, scalers, bank vectors, checkpoints and training histories.

## 11. Implementation milestones and acceptance gates

Implement in this order. A failed gate stops dependent production work but does not prevent writing/testing independent modules.

| Milestone | Deliverable | Required evidence |
|---|---|---|
| M0: contract and lineage | Architecture map; proposed config; data request; provider/quote/action/calendar decisions | `scope_freeze_draft.json`, unresolved-gate list; no claims of production readiness |
| M1: deterministic primitives | Schemas, hash/resume layer, synthetic calendar/action/label fixtures | Tests for every boundary/error case; legacy checkpoint rejection |
| M2: ingestion and features | Candidate snapshots; canonical adapter;23-column features; six fold manifests | Coverage/action/unit report; no future mutation; valid causal masks |
| M3: complete fresh training | Both backbones, ridge, validation selector, checkpoint resume | Training pilot with finite gradients and changed weights; artifact provenance |
| M4: memory and integration | Reference/optimized retrieval; random/plain/prior controls; gates; both selectors | Neighbor-ID parity, development-date audit, endpoint alias tests |
| M5: portfolio and analysis | Unified engine, metric functions, paired bootstrap and exports | Independent tiny hand-calculated ledgers; metrics/replay parity |
| M6: operational pilots | Laptop profile and VM-ready dependency inventory | `pilot_report.json`, timing projection, memory/disk peaks, interruption/recovery test |
| M7: push for review | Commit containing implementation, tests, config and pilot evidence | Exact full commit SHA; test output; unresolved risks; no production results required |
| M8: reviewed launch pack and final freeze | Reviewed config/data/code hashes; VM preflight and guarded launch commands | Review first authorizes VM preflight, not production. A30 runtime/backup gates and user approval then enable production. |

The companion acceptance inventory is the minimum suite, not a claim that tests exist or passed. Before pushing, complete the local tests and local pilot evidence; mark hardware-dependent A32–A34 as LOCAL_ONLY or PENDING_VM where appropriate. Run their A30 versions during reviewed VM preflight, before production authorization. Do not report pending VM checks as passed. No test may hard-code expected historical Sharpe as its success condition. A scientific run producing worse performance can pass every engineering test.

## 12. Restricted pilots and performance projection

### 12.1 Local pilots before consuming VM time

Use pre-2020 data and the 2020 fold's training/validation/development roles. No 2020–2025 performance selection during optimization pilots. Data integrity inspection is distinct from viewing policy outcomes.

1. Synthetic unit/integration tests first: corporate actions, exact session boundaries, missing bars, unavailable targets, aliasing, seed determinism, bankruptcy rejection and ledger reconciliation.
2. Train seed 7 of both models for three epochs on the 2020 training partition; validate only on 2018. Mark these pilot checkpoints ineligible for production. Check finite loss/gradients and actual weight updates; monotonic loss every epoch is not required.
3. Profile 20 warm-up optimizer steps followed by 100 timed steps, with device synchronization and separate loading/transfer/forward/backward timings. If the small partition is exhausted, repeat batches for timing only under a performance-fixture label, not scientific training.
4. Time 1000 development 2019 queries against the full available pilot bank. For latest-fold size stress, tile pre-2020 synthetic/real feature vectors under distinct performance-fixture IDs; never mistake that stress bank for evidence. Include cap/spacing selection and host/device transfers.
5. Run one complete 2019 development market account for every logical realization and mixture grid, then verify an independent ledger fixture. Measure serialized and bounded-parallel CPU versions; do not assume the GPU accelerates the engine.
6. Run 1000 bootstrap draws on synthetic or development returns, check replay and extrapolate to10000. Save both actual and projected time.
7. Interrupt one job after a completed optimizer step; resume; compare with uninterrupted execution on the same hardware. Corrupt a parent hash and confirm resume is refused.

### 12.2 Required pilot report

Record CPU model/core count, GPU model/dedicated VRAM, host RAM, driver/CUDA/PyTorch/yfinance versions, storage free/peak use, transfer throughput, batch settings, training samples per fold/market, steps per epoch, measured step/epoch times, retrieval queries/sec, simulation account-seconds, analysis draw-seconds, peak process RSS and GPU allocated/reserved memory, and failures. Include profiler traces; 30–40% GPU utilization alone does not identify the bottleneck.

Compute projected training time using 36 fits and the **50-epoch cap**, plus validation, not a hoped-for early-stop epoch. Add development grid simulations, 36 gates, retrieval,204 primary paths,96 cost-stress paths, inference, export and verification separately. Avoid charging shared forecasts/banks again for each policy. Pilot reporting may also give an early-stop scenario, clearly distinguished from the conservative cap projection.

Acceptance on a newly started24-hour VM: `1.5 * projected_remaining_compute + projected_export_and_verify + 2h_contingency <= actual_remaining_allocation`. Export-and-verify reserve is at least2 hours; reserve at least4 hours total after compute. At any elapsed time use the actual remaining budget, not a fresh24 hours. If the cap projection fails, do not launch all jobs and hope. Review the scope before outcome access, or finish checkpointed work on the laptop. Never drop losing seeds/markets or change stopping rules after seeing performance.

### 12.3 Resource ceilings and expiry behavior

The supplied VM image shows one A30 with 24GB dedicated VRAM, 16vCPUs, 64GB host RAM and 100GB disk. Treat disk as ephemeral until persistence is confirmed. Proposed operating ceilings: 48GB total application RAM, 20GB GPU allocation, 60GB working artifacts with at least 20GB free disk. Laptop ceilings: 10GB application RAM and 3.2GB GPU allocation on its 4GB dedicated device. Shared system memory is not an additional 4GB of physical GPU VRAM.

Storage calculations, not measurements:164000 float32 vectors of966 elements occupy633696000 bytes (0.634 decimal GB); materializing252-by-23 daily windows for the same records occupies3.802GB. A25000-by-164000 float32 distance array alone needs16.4GB. Therefore store one shared pooled bank, batch distances, retain selected neighbors, process folds sequentially and avoid seed-specific copies of input windows. Actual latest-fold row counts determine final sizing.

Target 20–40GB working data is a planning allowance, not a certified upper bound. The pilot measures physical usage including environments, caches, checkpoints, temporary outputs and archives. Pause safely when free disk falls below 20GB; do not delete unexported evidence to continue. Delete only regenerable caches whose parents and outputs are verified elsewhere.

Suggested reservation: hours 0–2 environment verification/A30 pilot;2–18 training and dependent prediction work;18–20 finish required analysis;20–22 export and verify;22–24 contingency. This is a deadline allocation, not a runtime promise. If data normalization or code implementation is unfinished, the24-hour allocation should not begin.

Export each completed fold's irreproducible/expensive outputs promptly; verify off-VM hashes. Keep latest resumable checkpoint backups at least hourly. The destination, credentials mechanism, actual expiry UTC and whether the clock is already running must be supplied before launch. Never embed secrets in the code or public pilot report. No machine is allowed to self-terminate before remote artifact verification.

## 13. What to push and what I will verify

Push the new namespace, tests, proposed/frozen configuration, dependency lock or exact environment inventory, architecture lineage, privacy-safe data manifest, all fold summaries, `pilot_report.json`, complete test results, and an artifact DAG sample. Provide the exact full commit SHA, not only 'latest'. Keep raw restricted data, credentials and large checkpoint blobs out of ordinary Git; provide their authorized access route and hashes separately.

The review will check actual training calls for both models; every representation path; date/label availability masks; data/action/unit semantics; retrieval constraints and boundary behavior; gate/selector isolation; event ordering and accounting; seeds/aliases/counts; metric and bootstrap consistency; resume and VM export behavior; and whether projected runtime fits the actual remaining allocation. Passing a six-test representation suite alone will not authorize the full scientific run.

Only after code review will the next handoff contain numbered acquisition/train/retrieve/evaluate/export commands, restart commands, VM-specific batching, and the cloud-to-local verification script. That pack starts with non-production VM preflight; its training entry point must refuse execution until the final freeze and hardware-dependent gates pass. The command names in this document describe interfaces to implement, not existing executable commands.

## 14. Required production decisions: no silent defaults

The draft JSON intentionally leaves the following unresolved: pinned implementation commit and dataset hashes; provider adjustment/action-basis verification; quote units and venue calendars; exceptional corporate-action/delisting treatment; architecture-lineage approval; data redistribution scope; actual VM expiry and persistence; verified off-VM destination; measured runtime and resource acceptance. Every field needs recorded evidence or an explicit scoped amendment. `production_authorized` remains false until all gates pass and the user approves the freeze.

A new full universe, added hierarchy module, alternative distance, revised horizon, target transformation, portfolio objective, cost policy or training search is a scientific amendment. Record it before evaluation output is opened. Pure performance improvements must preserve outputs within their declared numeric tolerances and pass reference parity; faster code is not permission to alter the tested method.

## 15. Source register and inspection limits

Sources below were inspected on 13 September2026. Public branch inspection is not an independent execution of the user's repository or data. At review time pin the pushed full commit and re-check these interfaces. The current public backbone module has legacy checkpoint loading and a 23-input MLP; the new architecture and fit loop in this plan deliberately replace those behaviors. The legacy universe file defines requested/retained identifiers, not point-in-time membership or verified execution metadata. Provider documentation exposes adjustment options and exclusive end dates; the profiler documentation supports measuring CPU/CUDA stages. The rest of this document is a proposed protocol, not a paraphrase of those sources.

```text
https://raw.githubusercontent.com/Rohil72/Core-RL-Agent/main/memory_study/backbones.py
https://raw.githubusercontent.com/Rohil72/Core-RL-Agent/main/memory_study/cache_builder.py
https://raw.githubusercontent.com/Rohil72/Core-RL-Agent/main/memory_study/engine_adapter.py
https://raw.githubusercontent.com/Rohil72/historical-memory-equity-data/v1.0.3/data/universe_manifest.csv
https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
```

No production training, market-data download, new manuscript results or VM scripts accompany this planning document.