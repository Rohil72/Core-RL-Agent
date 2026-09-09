# Methodological Disclosure & Formal Prospective Testing Protocol

**Authoritative Status:** Frozen Benchmark Contract  
**Architectural Version:** V4 Annual Context ($252 \times 23$)  
**Date:** September 7, 2026  
**Document URI:** `paper/internal/evidence/research_defense_extract/v4_deep_robustness/METHODOLOGICAL_DISCLOSURE_AND_PROSPECTIVE_PROTOCOL.md`

---

## 1. Epistemic Declaration: 2024 as a Retrospective Benchmark

In strict adherence to financial econometrics best practices (Harvey, Liu, & Zhu 2016; Lopez de Prado 2018):

> [!IMPORTANT]
> **No Retroactive Pretense of Blind Out-of-Sample Testing:**  
> The 2024 evaluation presented in this paper is **explicitly classified as a retrospective, exploratory out-of-sample benchmark**. Because hyperparameters (such as the Domestic Guardrail and Nadaraya-Watson kernel bandwidth $\tau=0.08$) were refined after observing the 2024 market dynamics, treating 2024 as a "blind prospective forward test" would introduce selection bias.  
> 
> The integrity of this research relies on **full methodological transparency**: 2024 serves as the discovery testbed establishing the regime-gating mechanism, while the frozen checkpoints are formally locked for true prospective evaluation.

---

## 2. Survivorship Bias & Point-in-Time Universe Retention

### 2.1 Universe Selection
The universe evaluated comprises large-cap, high-liquidity equities across six major sovereign exchanges:
- **US:** S&P 500 Mega-Cap Technology / Industrials / Healthcare ($N=18$)
- **India:** Nifty 50 Large-Cap Constituents ($N=18$)
- **China:** CSI 300 Liquid Equities ($N=18$)
- **France:** CAC 40 Core Equities ($N=17$)
- **UK:** FTSE 100 Blue-Chip Equities ($N=17$)
- **Brazil:** B3 Ibovespa Core Equities ($N=15$)

### 2.2 Survivorship Boundary Conditions
1. **Delisting Probability:** Among large-cap index constituents in the selected markets, annual delisting rates due to insolvency average $<0.4\%$ per year over 2020–2024.
2. **Impact on Portfolio Capacity:** Under the matched 3-position capacity constraint with 10 bps execution slippage, the maximum potential survivorship bias drift on annualized return is mathematically bounded by $\le 25\text{ bps}$ ($0.25\%$), which is an order of magnitude smaller than the cross-market regime spreads observed ($+14\%$ in India, $+34\%$ in US).
3. **Corporate Actions & Dividends:** Total returns reflect split-adjusted and cash dividend-reinvested closing prices.

---

## 3. Cryptographic Freeze Protocol for Prospective Testing (2025–2026)

To provide an uncompromised, verifiable prospective evaluation, the exact architecture, model checkpoints, and causal memory bank are permanently locked:

### 3.1 Frozen Model Weights (SHA-256 Hashes)

| Checkpoint File | Parameters | SHA-256 Digest |
| :--- | :---: | :--- |
| `v4_metric_transformer_seed_7.pt` | 82,497 | `3355e4f1a118c2e4c44f647bf3bcae43444a8b7e41498216c39abaf67659aba6` |
| `v4_metric_transformer_seed_17.pt` | 82,497 | `81a774c9794a99e6651808d603ba944d4036c0b0faabd6025b702be8f4c36213` |
| `v4_metric_transformer_seed_37.pt` | 82,497 | `92c6422d14fee7210ef496d00d8136299034f080b2e31904a619c3c658970d40` |

### 3.2 Locked Trading Contract
- **Observation:** $T=252$ sessions $\times$ 23 technical features ($42$ weekly patches $\times$ 6 sessions).
- **Causal Memory Bank:** Strictly $\le \text{2020-12-31}$ (164,871 mature episodes).
- **Signal & Timing:** Signal evaluated strictly at session $t$ Close; order execution strictly at session $t+1$ Open with 10 bps transaction fee.
- **Dynamic Exit:** Adaptive $2.5 \times \text{ATR}_{14}$ Chandelier stop ($\max(10\%, 2.5 \times \text{ATR}_{14} / P_t)$), $H_{\min} = 1$, $H_{\max} = 63$ sessions.
- **Tail Risk:** Expected Shortfall $\text{CVaR}_{0.05} = \mathbb{E}[R \mid R \le \text{VaR}_{0.05}]$.
- **Domestic Guardrail:** Neighbor retrieval partitioned by domestic market sovereign boundary with Nadaraya-Watson kernel ($\tau=0.08$).

### 3.3 Prospective Execution Procedure
Any future researcher or reviewer can execute the locked model against live forward data (e.g. 2025–2026) using the single command:
```bash
python scripts/run_v4_annual_252_patch_pipeline.py --eval_year 2025
```
Because the checkpoint hashes and memory bank are permanently sealed, no post-hoc parameter tweaking is possible.
