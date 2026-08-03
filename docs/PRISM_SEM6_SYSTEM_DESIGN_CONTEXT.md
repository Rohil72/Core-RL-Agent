# Prism AI Handoff: System Design and Methods Chapter

## Role in the Report

This is the second handoff for the Semester VI report. It follows
`PRISM_SEM6_REPORT_CONTEXT.md` and supplies Chapter 3 only.

Use this report order:

1. Introduction.
2. Literature Survey.
3. Materials and Methods: System Design.
4. Experimental Findings and Research Evolution.
5. Discussion, limitations, conclusion, appendices, and implementation detail
   are deferred to a later pass.

This is an Elsevier-style separation of concerns. The Introduction states the
objective and background. The Literature Survey positions prior work. The
Methods chapter must be reproducible and factual, with no result interpretation.
The Results chapter reports the measured outcomes without repeating the
architecture or literature review.

## Writing Rules for Chapter 3

- Use past tense for implemented experiments and present tense for the final
  design contract.
- State only components that were implemented. Label legacy, optional, and
  rejected branches explicitly.
- Do not repeat numerical performance values here. Refer forward to Chapter 4
  for results.
- Do not call the system a deployed trader, autonomous trading agent, or final
  RL agent.
- Do not describe news sentiment as an input. It was not implemented in the
  completed system.
- Use technical terms once, define them, and then use consistent names:
  encoder, latent state, decision adapter, market memory, evidence aggregator,
  deterministic policy, and backtest.
- Every figure must be self-explanatory, use editable LaTeX/TikZ vector
  elements, and be introduced in the surrounding text before it appears.
- Tables must be editable LaTeX tables, numbered in first-appearance order,
  have concise titles, avoid vertical rules and cell shading, and not duplicate
  values reported in a figure or later result table.

## Chapter Title and Opening

Use:

> Chapter 3: Materials and Methods - Retrieval-Grounded Market Memory System

Opening paragraph for Prism to adapt:

> This chapter describes the final research design as a modular causal decision
> system. The design was developed through successive ablations rather than
> assumed at the outset. Historical technical and point-in-time fundamental
> observations are encoded into a contextual market state, stored as
> outcome-matured historical experiences, and retrieved to form an interpretable
> distribution of future opportunity and risk. The final active path uses a
> deterministic long-only policy; cycle detection and offline reinforcement
> learning are retained only as earlier or rejected experimental branches.

## 3.1 Design Principles and Scope

State these four principles before describing modules:

1. **State before action.** The system first represents the market context; it
   does not directly map raw price changes to an action.
2. **Historical evidence before prediction.** A decision is grounded in
   outcome-matured analogues instead of relying solely on a parametric head.
3. **Causality before apparent performance.** A historical row becomes eligible
   memory only after its future horizon has fully elapsed.
4. **Modularity before optimisation.** Encoder, memory, evidence aggregation,
   and policy are separate so that one component can be tested without
   conflating it with another.

Do not call these universal guarantees. They are the design criteria used in
this research.

### Literature-to-Design Mapping

Introduce Table 3.1 here. It should map literature to design rationale, not
claim that the project reproduces any cited paper.

| Design decision | Literature motivation | Implemented interpretation |
|---|---|---|
| Patch-based temporal encoding | PatchTST and multivariate time-series representation learning | Encode 252 daily sessions with daily and patch-level Transformer streams. |
| Auxiliary masked reconstruction | Ti-MAE and representation-learning work | Use a low-weight reconstruction loss for encoder stability. |
| Retrieval-grounded reasoning | Retrieval-augmented forecasting and RETRO-style retrieval | Store historical latent states with later realized outcomes. |
| Distributional evidence | Sequential conformal and uncertainty work | Report empirical tails, disagreement, and evidence diagnostics; do not claim conformal coverage. |
| Selective / risk-aware participation | Calibration, selective prediction, volatility, and drawdown literature | Test confidence and risk branches; retain only transparent constraints in the final policy. |
| Multiple-testing control | PBO and Deflated Sharpe Ratio | Use promotion gates and distinguish development evidence from confirmation. |
| Offline RL comparison | CQL, IQL, TD3+BC | Treat offline RL as a tested comparator, not the final active policy. |

Suggested citations: `nie2023patchtst`, `zerveas2021transformer`,
`li2023timae`, `borgeaud2022retro`, `raft2025`, `raf2024`,
`xu2021conformal`, `zaffran2022adaptive`, `geifman2017selective`,
`bailey2014dsr`, `bailey2017pbo`, `kumar2020cql`, `kostrikov2022iql`, and
`fujimoto2021td3bc`.

## 3.2 Input Data and Causal Outcome Construction

Keep this section concise in the current report. Detailed data sourcing,
country-specific universes, and fundamental-data limitations belong in the
later methodology appendix.

### Inputs

Each observation is a daily company-level record containing 28 engineered
features over a 252-session lookback window:

- price returns, multi-horizon momentum, volatility, volume, intraday range,
  drawdown, trend, moving-average, and 52-week-relative technical features;
- point-in-time fundamental features related to earnings surprise, earnings
  growth, revenue growth, report availability, and time since the last report.

The encoder input tensor is `[batch, 28 features, 252 sessions]`. All
standardization statistics are fitted only on the applicable training split and
then reused for validation, test, and transfer inference.

### Future outcomes

Do not present the old oracle stage label as the final target. The final
decision research uses later realized outcomes, including:

- 21-, 63-, and 126-session returns;
- 63- and 252-session maximum upside;
- 63-session maximum downside;
- peak and drawdown timing offsets;
- upside-before-drawdown path events;
- fixed-horizon or counterfactual decision return, MFE, MAE, holding sessions,
  and benchmark-relative alpha.

For the final bounded rally memory, path quality is:

```text
positive(MFE) / (positive(MFE) + abs(negative(MAE)) + 1e-6)
```

It is bounded in `[0, 1]`. Explain that this repair prevents a nearly zero MAE
from creating an unbounded ratio that can dominate return evidence.

### Causal eligibility rule

For a memory item observed at time `t_m`, its outcome is not available to a
query at time `t_q` until the full evaluation horizon has matured. State the
rule abstractly:

```text
memory item is eligible only if outcome_available_timestamp <= query_timestamp
```

The implementation also excludes same-ticker matches and enforces a minimum
historical separation between neighbour and query dates. Timestamp comparisons
are normalized to nanosecond precision after a later audit found that mixed
timestamp precision could compromise causal filtering.

## 3.3 Temporal Transformer Market-State Encoder

This section should describe the active encoder without a training-history
digression.

### Architecture

| Element | Final specification |
|---|---|
| Lookback | 252 daily sessions |
| Input dimension | 28 features |
| Encoder family | Hierarchical patch Transformer |
| Token dimension | 96 |
| Daily Transformer | 4 layers, 4 attention heads |
| Patch construction | mean pooling over 5-session patches |
| Patch Transformer | 2 layers, 2 attention heads |
| Latent dimension | 128 |
| Internal memory | 8 slots of dimension 96, cross-attention read |
| Dropout | 0.10 |
| Activations | GELU in feed-forward and latent projections; LayerNorm at input and latent layers |

Describe the computation in prose:

1. The 28-dimensional daily feature vector is layer-normalized and projected to
   96 dimensions with learned positional information.
2. A four-layer daily Transformer models contextual session-level dynamics.
3. A second Transformer receives 5-session mean-pooled patch tokens, providing
   a coarser temporal view.
4. Attention pooling produces daily and patch summaries. These are fused with
   the latest daily state, latest standardized features, and an internal-memory
   cross-attention read.
5. A latent projection maps the fused 412-dimensional vector to a
   128-dimensional market state.

The encoder's internal memory is static during final inference. Runtime memory
updates are disabled to avoid batch-order-dependent representations.

### Encoder objectives

The active final encoder objective is masked Huber regression over the 11
future outcomes plus low-weight masked reconstruction:

```text
L_encoder = L_masked_huber(future outcomes) + 0.02 * L_masked_reconstruction
```

Legacy or ablation configurations also evaluated ranking, analogue geometry,
VICReg-style variance regularization, supervised contrastive, and triplet
terms. Those are not simultaneously claimed as active final losses. One loss
sweep discovered an instability caused by latent-vector normalisation at very
small norms; using `eps = 1e-4` bounded the gradient scale.

## 3.4 Optional Decision Adapter

The decision adapter is a separate, lightweight mapping from the frozen
128-dimensional encoder state to a 32-dimensional retrieval space:

```text
LayerNorm(128) -> Linear(128, 64) -> GELU -> Dropout(0.10) -> Linear(64, 32)
-> L2 normalization with eps = 1e-4
```

Its role is not to replace market memory. It tests whether a decision-oriented
representation makes retrieval geometry more useful than the raw encoder
latent. The report must describe raw and adapter retrieval as explicitly
separate embedding views. A later audit found that a generic loader could
silently select raw `latent_*` columns when both raw and adapter columns were
present. The final study materializes exactly one embedding family per view and
checks its dimension and provenance.

Do not say the adapter is universally superior. The results show mixed transfer
performance.

## 3.5 Causal Historical Market Memory

Each memory item stores:

```text
embedding, ticker, market, date, outcome availability time,
future upside, relative alpha, future downside, holding period,
path quality, and retrieval metadata
```

For a query state, the memory layer:

1. filters ineligible future information and prohibited same-ticker neighbours;
2. identifies the closest historical states in the chosen raw or adapter space;
3. limits repeated evidence from a single ticker when a ticker cap is enabled;
4. aggregates evidence with Gaussian distance weighting rather than a simple
   unweighted neighbour mean;
5. returns expected upside, alpha, downside, downside tail, holding period,
   path quality, neighbour count, effective sample size, entropy, diversity,
   distance, agreement, and multi-scale disagreement.

The standard configuration retrieves 25 neighbours, requires at least 15,
uses a 126-session outcome-maturity horizon, and requires 21 sessions of
historical separation. The bounded multiscale variant evaluates evidence at
10, 25, and 50 neighbours and penalizes disagreement among those scales.

Use the phrase "empirical predictive tail" rather than "confidence interval"
when discussing the lowest retrieved-outcome percentile. The system does not
yet establish formal calibration.

## 3.6 Deterministic Opportunity Policy

The active policy is deterministic and long-only. It ranks eligible equities by
retrieval evidence and selects at most three positions. It does not learn an
action through online interaction.

The policy uses transparent constraints:

- expected upside and alpha evidence must be positive;
- expected downside and downside-tail limits must remain acceptable;
- a minimum evidence-confidence and neighbour-count condition applies;
- positions have a 5-session minimum hold, 63-session maximum hold, 10% stop,
  and 10 basis points of slippage per side in decision-outcome construction;
- seed-specific rankings are aggregated continuously by mean percentile rank.

The final active study sets zero mandatory binary seed votes. Do not describe it
as a two-vote consensus system; that belongs to an earlier locked policy branch.

Risk guards, cooldowns, and drawdown constraints are transparent rules around
the ranking policy. They are not presented as a learned exposure controller.

## 3.7 Experimental Branches and Design Evolution

This section is important because the final design cannot be understood without
the results that selected it. Introduce the evolution table in Methods, but
reserve all performance numbers and interpretation for Chapter 4.

| Component or hypothesis | Why it was explored | Final treatment | Results section link |
|---|---|---|---|
| Oracle Stage 1-4 detector | Early interpretable market-cycle framing | Legacy weak-supervision tool; not final state definition | Sec. 4.2 |
| Direct feed-forward head | Baseline prediction mechanism | Comparator only; retrieval was more useful in early evidence | Sec. 4.3 |
| Outcome-aware latent losses | Improve analogue geometry | Diagnostic ablation; stable training repair retained | Sec. 4.3 |
| Raw latent memory | Direct historical analogue retrieval | Retained as C0 control | Sec. 4.7 |
| Decision adapter | Align representation with decision outcomes | Retained as optional explicit retrieval view | Sec. 4.5 and 4.7 |
| Exposure controller | Reduce volatility and drawdown | Not active; risk reduction did not create robust alpha | Sec. 4.4 |
| Reliability / abstention filter | Avoid untrustworthy memory states | Rejected after poor transfer and sealed confirmation | Sec. 4.4 and 4.6 |
| Rally and neutral prototype banks | Detect favorable entry paths | Informed rally outcomes; binary gate not retained | Sec. 4.4 |
| Offline RL | Learn allocation from fixed data | Rejected tested formulation; not active | Sec. 4.6 |
| Global / regional memory topology | Test cross-market transfer | Frozen global source used in closing diagnostic; no universal winner claimed | Sec. 4.6 |
| Bounded path quality and coverage audit | Repair final memory validity defects | Required in closing diagnostic | Sec. 4.7 |

## 3.8 Evaluation Integrity and Reproducibility

This section should be factual, compact, and placed at the end of Methods.

- Markets studied: US, India, China, Brazil, France, and UK.
- Seeds: 7, 17, and 37 where the relevant experiment used seed ensembles.
- Report periods: development (2022-2023), selection (2024), and observed
  diagnostic period (2025 through 2026 Q1).
- The observed period was inspected during development. It is diagnostic only,
  not fresh confirmation evidence.
- Baselines include equal-weight buy-and-hold and momentum under the same
  execution contract where available.
- Robustness gates assess market wins, drawdown, concentration, seed lift, PBO,
  and Deflated Sharpe rather than accepting a pooled Sharpe alone.
- The durable runner records immutable input/source fingerprints, expected
  artifacts, checkpoints, and runtime information. This supports restart-safe
  execution but does not itself prove economic validity.

State the evidence boundary explicitly:

> Fixed liquid-company universes retain survivorship and researcher-selection
> bias. The report therefore presents the work as a causal and reproducible
> research framework, not as a deployable investment product.

## Exact Diagram Instructions for Prism

### Figure 3.1 - Final System Architecture

**Purpose:** Make the final active data flow understandable at one glance.

**Format:** Draw as a landscape TikZ vector figure, approximately `0.95\textwidth`,
with no gradients, no decorative graphics, no stock-price artwork, and no
screenshots. Use a white background, black/charcoal text, muted teal for retained
components, muted grey for data stores, and thin muted red outlines only for
explicitly rejected legacy branches. Use a consistent sans-serif font at a
legible final PDF size.

**Layout, left to right:**

1. `Daily technical + point-in-time fundamental features` with sublabel
   `28 features x 252 sessions`.
2. `Patch Transformer market-state encoder` with sublabel
   `daily stream + 5-session patch stream; 128-D latent`.
3. `Explicit retrieval view` split into two vertically stacked small boxes:
   `Raw state: 128-D` and `Optional decision adapter: 32-D`.
4. `Causal historical market memory` with sublabel
   `outcome-matured analogues only; same ticker excluded`.
5. `Distributional evidence aggregation` with sublabel
   `upside, alpha, downside tail, path quality, agreement`.
6. `Deterministic top-3 long-only policy` with sublabel
   `risk limits, hold/exit rules, seed-rank aggregation`.
7. `Backtest and robustness gates` with sublabel
   `return, Sharpe, drawdown, concentration, PBO`.

Use solid arrows for the forward information flow. Add one dashed arrow from
the later realized-outcome store back into historical memory only, labelled
`admitted after outcome maturity`. Do not draw an arrow from future outcomes to
the current query or policy. Place a small dashed, grey side note below the
main chain: `Legacy/rejected: cycle detector, reliability filter, offline RL`.

**Caption:**

> Figure 3.1. Final modular retrieval-grounded market-memory design. A query
> state is compared only with historical experiences whose future outcomes had
> already matured at the query date; the active decision policy is deterministic
> and long-only.

### Figure 3.2 - Causal Memory Eligibility Timeline

**Purpose:** Explain the most important validity rule without a dense formula.

**Format:** A horizontal timeline across `0.90\textwidth`. Use two aligned
rows: one for a historical memory item and one for the current query. Render it
as TikZ, not as a bitmap.

**Top row:**

- tick at `t_m`: `historical state encoded`;
- shaded interval from `t_m` to `t_m + H`: `future outcome horizon`;
- tick at `t_m + H`: `outcome becomes available`;
- arrow to a later query time `t_q` with condition `t_m + H <= t_q`.

**Bottom row:**

- tick at `t_q`: `current state queried`;
- arrow from eligible historical embedding to retrieved-neighbour set;
- a crossed-out arrow from a non-matured item with label `excluded`.

**Caption:**

> Figure 3.2. Causal eligibility of a historical analogue. A state may enter
> memory only after the future horizon used to evaluate that state has elapsed.

### Figure 3.3 - Design Evolution from Early Hypotheses to Final Active Path

**Purpose:** Make the project continuation coherent and show that rejected
branches informed the final design.

**Format:** A vertical three-column flow diagram, `0.95\textwidth`, built with
TikZ. Avoid a chronological spaghetti graph.

**Columns:** `Early framing`, `Tested refinement`, `Final active / status`.

**Rows:**

1. `Stage/cycle detector` -> `latent-state audit` -> `latent opportunity state
   retained; detector legacy`.
2. `Direct prediction head` -> `historical analogue retrieval` -> `memory
   retained; head comparator`.
3. `Simple neighbour mean` -> `distributional evidence and multi-scale
   agreement` -> `bounded rally evidence under diagnostic repair`.
4. `Risk/exposure heuristics` -> `confidence and reliability branches` ->
   `risk insights retained; learned filter rejected`.
5. `RL allocation hypothesis` -> `CQL/IQL/TD3+BC comparison` -> `tested
   formulation rejected; deterministic policy retained`.

Use green/teal only for retained outcomes, grey for neutral experimental steps,
and red outline with the word `rejected` for failed branches. Do not show Sharpe
figures inside the graphic; those belong in Chapter 4.

**Caption:**

> Figure 3.3. Design evolution guided by ablation evidence. Rejected branches
> are reported as experimental findings and are not represented as active
> components of the final system.

### Tables for Chapter 3

Include only these three tables:

1. **Table 3.1. Literature-to-design mapping.** Use the compact table in
   Section 3.1.
2. **Table 3.2. Final component contract.** One row per retained module:
   encoder, raw/adapter retrieval view, causal memory, evidence aggregator,
   deterministic policy, evaluation runner. Columns: input, output, fixed
   contract, and interpretability artifact.
3. **Table 3.3. Design evolution and final disposition.** Use Section 3.7.

Do not make a table of experimental performance in Chapter 3. Those belong in
Chapter 4 and should not be duplicated.

## What Must Remain for the Next Pass

Do not generate these items yet:

- full data-universe and ticker list;
- country-specific data collection detail;
- full loss-sweep appendix;
- detailed backtest equations and transaction-cost sensitivity;
- architecture implementation listings;
- results charts, equity curves, heatmaps, or trade examples;
- Discussion and conclusion prose.

Those depend on the final selected reporting figures and on the closing memory
repair result, which is not yet reportable as a completed experiment.
