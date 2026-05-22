# Current Architecture

This project is a batch equity-cycle modeling pipeline. The active model is a
hierarchical LSTM sequence model trained on daily technical and report-aware
fundamental features. The old PPO/RL stack is not part of the active path.

## End-To-End Flow

```text
Ticker universe
  -> raw OHLCV + earnings/revenue fetch
  -> feature engineering
  -> weak cycle detector
  -> oracle/action metadata + price-derived event targets
  -> rolling sequence dataset
  -> hierarchical LSTM cycle model
  -> future/event prediction + weak action prediction
  -> cycle-span evaluation
```

## Inputs

The raw input universe is defined in:

```text
config/market_universe.yaml
```

For each ticker, the loader fetches:

- daily OHLCV bars: `open`, `high`, `low`, `close`, `volume`
- earnings report dates
- EPS estimate and reported EPS
- quarterly revenue when available

The default runtime config is:

```text
configs/cycle_model.yaml
```

## Preprocessing

Preprocessing is handled mainly by:

```text
src/data/loader.py
src/data/features.py
scripts/precompute_ground_truth.py
```

The preprocessing stage builds a daily dataframe per ticker.

### Technical Features

The current sequence feature set has 28 inputs. Technical features include:

- daily return
- intraday range
- momentum over `3`, `10`, and `21` days
- 21-day volatility
- volume ratio and daily volume change
- drawdown from rolling high
- short trend slope
- close versus `50`, `150`, and `200` day SMAs
- 200-day SMA trend
- position versus 52-week low/high
- up/down volume ratio
- Minervini-style template score and gate

### Fundamental Features

Fundamentals are aligned to price dates without lookahead. Features include:

- EPS surprise
- EPS growth year over year
- rolling two-quarter EPS growth
- EPS acceleration
- revenue growth year over year
- rolling two-quarter revenue growth
- Minervini-style fundamental score
- report availability flag
- days since last report

The model input tensor is:

```text
[batch, features, time] = [B, 28, 252]
```

The model internally transposes this to:

```text
[B, 252, 28]
```

## Target Construction

Targets are built in:

```text
src/cycle/cycle_detector.py
src/cycle/oracle.py
```

### Weak Detector Targets

The cycle detector produces heuristic cycle spans. These are useful, but they
are treated as weak labels rather than ground truth.

The oracle layer converts detected spans into:

- `oracle_action`
- `oracle_cycle_id`
- cycle start/end/peak indices
- cycle start/end/peak dates
- cycle duration and progress
- enter/exit flags

The action classes are:

```text
0 = neutral
1 = enter
2 = hold_or_renew
3 = exit
```

### Price-Derived Outcome Targets

The future/outcome target vector currently has 11 dimensions:

```text
future_return_21
future_return_63
future_return_126
future_return_252
future_max_return_63
future_min_return_63
event_peak_offset_63
event_drawdown_offset_63
event_upside_before_drawdown_126
event_upside_hit_126
event_drawdown_hit_126
```

The first six are future return/extreme-return targets. The last five are
event-style targets derived directly from future prices, not from the detector.
They encode:

- time to best future close inside 63 days
- time to worst future close inside 63 days
- whether upside happens before drawdown inside 126 days
- whether upside is hit inside 126 days
- whether drawdown is hit inside 126 days

This is the main shift away from pure detector imitation.

## Dataset

The rolling dataset is built in:

```text
src/data/sequence_dataset.py
```

Each sample is a 252-day window ending at the prediction date:

```text
sequence:      [28, 252]
action:        scalar class in {0, 1, 2, 3}
future_target: [11]
```

Splitting is walk-forward:

```text
3 years train / 1 year validation / 1 year test
```

The default ticker holdout fraction is:

```text
0.20
```

Feature standardization is fitted on the training split only.

## Model

The active model is:

```text
src/models/hierarchical_lstm_model.py
HierarchicalLSTMCycleModel
```

## World-Model Framing

The current architecture is best understood as a small predictive world model
for equity-cycle state, not just an action classifier.

```text
past 252 trading days
  -> observation encoder
  -> recurrent memory
  -> latent market state
  -> predict future outcomes, event timing, risk ordering, and weak actions
```

In this framing:

- the input sequence is the model's observation history
- the LSTM stack is the memory system
- the `latent` vector is the learned belief/state of the market setup
- the future/event head is the outcome model
- the reconstruction head is the observation model
- the action head is a weak policy/readout trained from detector labels

This is not yet a full generative simulator. It does not roll forward synthetic
OHLCV paths step by step. But it already learns a compact state that is trained
to explain the past and predict future market consequences. That is the useful
part of a world model for this project.

Default model config:

```yaml
window_size: 252
encoder: hierarchical_lstm
lstm_hidden_dim: 96
lstm_layers: 2
patch_size: 5
latent_dim: 128
dropout: 0.10
```

## Memory System

The memory design is hierarchical.

### Short-Term Daily Memory

The first memory level reads every trading day in the 252-day window.

```text
[B, 252, 28] -> daily LSTM -> [B, 252, 96]
```

This layer is responsible for local sequence information:

- short momentum shifts
- report-date reactions
- volatility clusters
- pullbacks
- volume changes
- recent trend-template changes

Because it is recurrent, the hidden state at each day carries information from
all previous days in the window.

### Patch-Level Stage Memory

Daily states are grouped into 5-day patches:

```text
252 trading days / 5 days ~= 51 patch tokens
```

Then a second LSTM runs over the patch sequence:

```text
[B, 51, 96] -> patch LSTM -> [B, 51, 96]
```

This layer is meant to learn slower stage structure:

- base formation
- early breakout
- post-breakout confirmation
- trend continuation
- late-stage exhaustion
- risk/drawdown setup

This is where the model gets a more monthly/stage-wise view instead of treating
all 252 daily steps equally.

### Attention Memory Read

The model does not rely only on the final LSTM state. It performs attention
pooling over both memory streams:

```text
daily attention context = weighted read over 252 daily states
patch attention context = weighted read over 51 patch states
latest daily state      = final recurrent state
latest raw features     = current-day observation
```

This gives the model three different memory reads:

- what mattered anywhere in the full daily history
- what mattered across slower stage patches
- what the most recent state says right now

The final latent state is therefore a fused belief state, not just a last-day
embedding.

### Model Shape

Input:

```text
[B, 28, 252]
```

Transpose:

```text
[B, 252, 28]
```

Input projection:

```text
LayerNorm(28)
Linear(28 -> 96)
GELU
LayerNorm(96)
```

Daily LSTM:

```text
2-layer LSTM
input size = 96
hidden size = 96
output = [B, 252, 96]
```

Patch construction:

```text
patch_size = 5 trading days
252 days -> ceil(252 / 5) = 51 patch tokens
patch tokens = [B, 51, 96]
```

Patch LSTM:

```text
1-layer LSTM
input size = 96
hidden size = 96
output = [B, 51, 96]
```

Attention pooling:

```text
daily attention context = [B, 96]
patch attention context = [B, 96]
latest daily state      = [B, 96]
latest raw features     = [B, 28]
```

Fused vector:

```text
[B, 96 + 96 + 96 + 28] = [B, 316]
```

Latent state:

```text
Linear(316 -> 128)
GELU
LayerNorm(128)
Dropout(0.10)
latent = [B, 128]
```

The `latent` vector is the central memory-compressed market state. It is the
representation every downstream prediction depends on.

### Output Heads

Future/event head:

```text
Linear(128 -> 128)
GELU
Linear(128 -> 11)
future_pred = [B, 11]
```

This head makes the latent state predictive. It is trained to answer questions
such as:

- what is the likely 21/63/126/252-day return?
- what is the best upside inside 63 days?
- what is the worst drawdown inside 63 days?
- how far away is the future peak?
- how far away is the future drawdown?
- does upside happen before drawdown?

That is the current world-model core: the latent state must encode enough about
the market setup to predict future consequences.

Action head:

```text
concat(latent, future_pred) = [B, 139]
Linear(139 -> 128)
GELU
Dropout(0.10)
Linear(128 -> 4)
action_logits = [B, 4]
```

This head is deliberately treated as weaker than the future/event head. It is
still trained from detector-derived action labels, but the detector is no longer
the only learning signal.

Masked reconstruction head:

```text
Linear(96 -> 28)
reconstruction = [B, 28, 252]
```

This is the observation-model side of the architecture. During training, parts
of the input sequence are hidden, and the model must reconstruct them from
context. This forces the memory system to learn relationships among technical
features, fundamental/report features, and time.

The normal inference contract is:

```python
{
    "latent": latent,
    "future_pred": future_pred,
    "action_logits": action_logits,
}
```

During training, reconstruction can also be requested:

```python
{
    "latent": latent,
    "future_pred": future_pred,
    "action_logits": action_logits,
    "reconstruction": reconstruction,
}
```

## Training Losses

Training is handled by:

```text
src/trainers/train_cycle_model.py
```

Current losses:

```text
total_loss =
  action_loss_weight * weighted_action_cross_entropy
  + future_loss_weight * smooth_l1(future_pred, future_target)
  + reconstruction_loss_weight * masked_reconstruction_loss
```

Default weights:

```yaml
action_loss_weight: 0.75
future_loss_weight: 0.35
reconstruction_loss_weight: 0.10
```

Hard-negative rows increase the action loss weight:

```text
row_weight = 1.0 + hard_negative * hard_negative_weight
```

Default:

```yaml
hard_negative_weight: 1.0
```

### Self-Supervised Masking

During training, the model masks standardized sequence inputs and reconstructs
only the masked positions.

Default masking config:

```yaml
mask_probability: 0.10
mask_span_probability: 0.03
mask_span_length: 5
mask_value: 0.0
```

This encourages the LSTM to learn robust temporal and cross-feature structure
instead of only memorizing detector labels.

## What Is Novel Here

The current model is not just a stock LSTM classifier. The novel pieces are:

- hierarchical recurrence over both daily states and short trading-week patches
- attention pooling over daily and patch-level histories
- fusion of latest raw features with long-context recurrent state
- joint prediction of future returns, event timing, risk ordering, and weak cycle actions
- detector labels are downgraded to a weighted auxiliary signal rather than treated as the only supervision
- masked reconstruction adds a self-supervised representation objective

The research direction is to make the latent representation learn cycle/stage
structure from realized market paths, while using the detector only as a weak
proposal mechanism.

## Outputs

Training writes:

```text
models/cycle_reasoner/final_model.pt
reports/experiments/*.json
reports/experiments/*.md
```

Evaluation writes:

```text
reports/evaluations/*.json
reports/evaluations/*.md
data/evaluation_plots/*_comparison.png
```

The numeric evaluation reports include:

- action accuracy, balanced accuracy, macro/weighted F1
- action confusion matrix, prediction distribution, and per-class support
- future/event MAE, RMSE, mean R2, and mean Pearson correlation
- per-target future/event regression metrics
- cycle precision, recall, F1, return statistics, catastrophic rate, and exit error

Generated outputs are ignored by git.
