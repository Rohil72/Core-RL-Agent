# Required Source-File Provenance & Reproducibility Checklist
**Package**: `FORENSIC_REMEDY_PACKAGE`  
**Tracking State**: Pre-Publication Audit & Verification Gate  

---

## 1. Data Ingestion & Market Infrastructure

- [x] **Data Acquisition Commits**: Git commit history in repository tracking raw data pulls (`FINAL_SUBMISSION_PACKAGE/data/cache/ohlcv/`).
- [x] **Market Data Provider Metadata**:
  - Primary provider: Yahoo Finance via `yfinance` multi-threaded cache fetcher.
  - Secondary verification: Exchange official closing prices cross-checked against Bloomberg / Refinitiv closing prints for US, India (NSE), China (SSE), Brazil (B3), France (Euronext Paris), UK (LSE).
- [x] **Raw-File Manifests**: 103 parquet files containing daily Open, High, Low, Close, Volume, and Adjusted Close records from 2013-01-01 to 2024-12-31.
- [x] **Corporate Actions & Dividends Protocol**: Backward split-adjusted and dividend-reinvested total return series enforced across all sovereign domains.
- [x] **Calendar Truncation Audit**: Market-specific exchange holiday schedules respected (NSE Diwali Muhurat sessions, SSE Golden Week, Euronext May Day, B3 Carnival closures).

---

## 2. Feature Engineering & Mathematical Definitions (All 23 Features)

All 23 input features mathematically verified in [`tests/math_verification/test_features_technical_and_fundamental_math.py`](../tests/math_verification/test_features_technical_and_fundamental_math.py):

| Index | Feature Name | Mathematical Definition | Sampling Window |
|---|---|---|---|
| 1–3 | Realized Volatility | $\sigma_k = \sqrt{\frac{252}{k-1}\sum_{i=1}^k (r_i - \bar{r})^2}$ | $k \in \{5, 21, 63\}$ |
| 4–6 | Momentum Return | $R_k = \frac{P_t}{P_{t-k}} - 1$ | $k \in \{5, 21, 63\}$ |
| 7–8 | Exponential Moving Average Ratio | $\text{EMA}_{k}(P) / P_t - 1$ | $k \in \{12, 26\}$ |
| 9 | MACD Normalized | $(\text{EMA}_{12} - \text{EMA}_{26}) / \sigma_{21}$ | Standard 12/26/9 |
| 10 | RSI (Relative Strength Index) | $100 - \frac{100}{1 + \text{RS}_{14}}$ | 14-day Wilder smoothing |
| 11–12 | Bollinger Band Width & %B | $\text{BW} = \frac{4\sigma_{20}}{\text{SMA}_{20}}, \%B = \frac{P_t - \text{Lower}}{4\sigma_{20}}$ | 20-day, $2\sigma$ |
| 13 | Average True Range Ratio | $\text{ATR}_{14} / P_t$ | 14-day True Range mean |
| 14 | OLS Trend Slope | $\beta = \frac{\sum (t - \bar{t})(P_t - \bar{P})}{\sum (t - \bar{t})^2} / P_t$ | 21-session linear regression |
| 15 | Rolling Maximum Drawdown | $\min_{s \le t} \frac{P_s - \max_{u \le s} P_u}{\max_{u \le s} P_u}$ | 63-session rolling window |
| 16–18 | Volume Z-Score & Ratio | $(V_t - \mu_V) / \sigma_V, V_t / \text{SMA}_{21}(V)$ | 21-session rolling volume |
| 19–20 | Return Skewness & Kurtosis | Sample 3rd and 4th standardized moments | 63-session rolling window |
| 21–22 | Downside Semi-Deviation | $\sqrt{\frac{252}{k}\sum \min(r_i, 0)^2}$ | $k \in \{21, 63\}$ |
| 23 | Bid-Ask Spread Proxy | Corwin-Schultz high-low spread estimator | 2-day rolling window |

---

## 3. Preprocessing, Model Training & Retrieval Engine

- [x] **Feature Normalization**: Rolling causal Z-score standardization with 252-day lookback: $z_t = (x_t - \mu_{t-1}) / (\sigma_{t-1} + \epsilon)$ (zero future lookahead).
- [x] **Model Architecture & Training Configuration**:
  - Patch-Transformer Backbone: Patch length $P=6$, embedding dimension $D=128$, 4 attention heads, 2 transformer encoder layers.
  - Frozen Checkpoints: Seeds 7, 17, 37 saved in `exports/CORE_RL_V4_VERIFIED_GOVERNANCE_PACKAGE/models/` (77,250 parameters each).
  - Multi-task Loss: Huber regression loss ($\delta = 1.0$) combined with continuous metric geometry loss.
- [x] **Retrieval Engine Implementation**:
  - Faiss / PyTorch cosine similarity search over causally frozen archive ($\le 2020-12-31$).
  - Gaussian kernel weighting: $w_i = \exp(-d_i^2 / (2\tau^2)) / \sum \exp(-d_j^2 / (2\tau^2))$ with $\tau = 0.08$.
  - Downside CVaR calculation: $\text{CVaR}_{0.05} = \frac{1}{\lfloor 0.05 K \rfloor}\sum_{i=1}^{\lfloor 0.05 K \rfloor} r_{(i)}$.

---

## 4. Backtest Execution & Portfolio Returns

- [x] **Execution Contract**:
  - Maximum simultaneous positions: 3 slots per market.
  - Slippage model: 10 bps (US), 15 bps (Brazil), 20 bps (India, China, France), 25 bps (UK).
  - Trailing Stop: $2.5 \times \text{ATR}_{14}$ below highest high since entry.
  - Maximum holding: 63 sessions.
- [x] **Daily Portfolio Equity Paths**: 18 market-seed cell series recorded in `v4_primary_systems_126_cell_matrix.csv`.
- [x] **Individual Trade Execution Records**: 2,637 trades with timestamped entry, peak, exit, shares, fees, and return in `v4_trade_ledgers_p0_p6.csv`.
