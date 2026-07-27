# Architecture

## Active System

The active system learns a market state from history, retrieves outcome-matured
historical analogues, turns them into a distribution of evidence, and evaluates
exposure decisions. It is no longer a detector-centric cycle classifier.

```text
feature history -> patch Transformer -> latent state -> causal memory
-> evidence distribution -> deterministic consensus/reliability policy
-> backtest and gates
```

## Data and Targets

`scripts/precompute_ground_truth.py` produces per-ticker Parquet frames. The
final encoder consumes 28 daily technical and point-in-time fundamental features
over 252 sessions: `[B, 28, 252]`. Training-only standardization prevents split
leakage.

The 11 future targets are 21/63/126-session returns; 63/252-session upside;
63-session downside; peak/drawdown offsets; upside-before-drawdown; upside-hit;
and drawdown-hit. Detector action targets remain available for legacy work, but
the final configuration disables them.

## Patch Transformer

`src.models.patch_transformer_model.HierarchicalPatchTransformerCycleModel` is
the active encoder:

```text
input: [B, 28, 252] -> [B, 252, 28]
LayerNorm(28) -> Linear(28, 96) + learned position
4 daily Transformer layers, 4 heads, GELU feed-forward blocks
mean pooling into 5-session patches
2 patch Transformer layers, 2 heads
attention pool over daily and patch streams
```

The latent head fuses daily context [96], patch context [96], latest daily
state [96], latest raw features [28], and internal memory read [96].
`Linear(412, 128) -> GELU -> LayerNorm -> Dropout(0.10)` produces the latent.
The future head is `128 -> 128 -> 11`; the compatibility action head is
`(128 + 11) -> 128 -> 4`; reconstruction maps daily states `96 -> 28`.

The encoder has eight 96-dimensional internal memory slots read by four-head
cross-attention. Final-testbed runs use `static_parameter` memory with no EMA
updates, avoiding batch-order-dependent runtime memory.

## Encoder Objective

The final encoder uses masked Huber future regression plus `0.02` weighted
masked reconstruction. Action loss is zero. Optional loss-sweep configurations
can enable regression, analogue, ranking, variance, supervised-contrastive, or
triplet terms through `training.loss`; these terms are not active by default.
L2 normalization in the geometry loss uses `eps=1e-4` to cap unstable gradients
from very small latent norms.

## Historical Market Memory

`src/memory/` stores each frozen latent with identity metadata and later
outcomes. The final decision memory uses maximum favorable excursion, net
relative alpha, maximum adverse excursion, path quality, and holding sessions.

For each query, it normalizes latents using memory statistics, filters illegal
neighbours, and retrieves 25 analogues. The final testbed excludes same-ticker
matches, uses a 126-session causal horizon and 21-session separation, and can
filter on sector, industry, distance, confidence, and outcome availability.

Gaussian aggregation returns expected upside/alpha/downside, quantiles,
confidence intervals, tail risk, holding period, and path quality. Confidence
uses neighbour agreement/disagreement, effective sample size, entropy, distance,
and historical diversity. The testbed scores opportunities by alpha lower
confidence bound rather than an unweighted neighbour mean.

## Decision and Policy Layer

`src/decision/dataset.py` derives later-price counterfactual outcomes from
frozen states: fixed-horizon return, MFE, MAE, path quality, benchmark-relative
net alpha, and utility after slippage. The final protocol uses a 63-session
primary horizon and 10 bps per side. When benchmark alpha is unavailable, it
uses a local cross-sectional median fallback.

The final policy is deterministic. It combines three seed-specific retrieval
runs, requires two votes, ranks opportunities by consensus evidence, and applies
a chronologically fitted reliability model. The locked candidate uses the 25%
reliability threshold, top-three allocation, downside and CVaR limits, a
63-session maximum hold, a 10% stop, and the configured market execution costs.

`src/policy/offline_policy.py` and the contextual-bandit/CQL/IQL/TD3+BC scripts
remain for reproducibility. Phase 6 showed that those learned exposure policies
did not preserve the memory signal consistently across markets, so none is part
of the final architecture.

## Integrity and Evaluation

The durable runner fingerprints source, inputs, expected outputs, Python,
PyTorch, CUDA, driver, and accelerator class. Encoder continuation checkpoints
include model, optimizer, AMP scaler, RNG state, epoch/batch progress, and
runtime/config fingerprints.

Development selection requires market wins, non-degenerate exposure, positive
median excess Sharpe, bounded drawdown, and bounded PBO. Confirmation can only
be compiled after promotion; it locks source, data, models, and policy datasets
before evaluating the untouched candidate across 2025 through 2026 Q1.

`configs/final_memory_policy.yaml` is the frozen architecture contract: global
transformer, global internal memory, global external memory, no RL, and no
further development-threshold search.
