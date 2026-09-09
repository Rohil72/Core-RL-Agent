# Prospective Evaluation Registration & Pre-Specification Protocol

**Date of Public Registration & Checkpoint Freeze:** September 9, 2026  
**Git Commit Hash (HEAD):** `dc09c633101569cc6c40b5bf09add255b68b9e23`  
**Prospective Evaluation Period:** Calendar Year 2027 (January 1, 2027 – December 31, 2027)  
**Primary Evaluation Topology:** $P_0^*$ (Annual Causal Patch Transformer + Nadaraya-Watson Retrospective Retrieval + Volatility/CVaR Guardrail)  
**Parametric Comparator System:** $P_1$ (Ablated Transformer without Episodic Memory Bank)  
**Authoritative Artifact Directory:** `exports/CORE_RL_V4_FAITHFULNESS_AND_FREEZE_PACKAGE/`  

---

## 1. Executive Summary & Research Identity

This registration protocol publicly freezes the model architecture, trained neural network weights, causal memory bank, retrieval parameters, portfolio execution logic, and statistical evaluation criteria for the Core-RL framework prior to observing any prospective data from the evaluation window (Calendar Year 2027).

The central scientific claim of this research is **not** that retrospective memory engineering produces unboundedly superior speculative alpha over modern deep architectures. Rather, it addresses an institutional research question:
> *Can an interpretable, case-based retrospective retrieval policy deliver equivalent, non-inferior risk-adjusted performance compared to opaque parametric benchmarks while providing point-in-time, auditable historical precedents for every individual capital allocation?*

To protect against hindsight optimization, lookahead bias, and researcher degrees of freedom, all parameters, seeds, weights, and decision rules are frozen as of **September 9, 2026** at Git commit `dc09c633101569cc6c40b5bf09add255b68b9e23`.

---

## 2. Frozen Model Weights & Cryptographic Checksums

The pre-trained transformer encoders were trained strictly on mature market regimes concluding on or before December 31, 2020. The model weights are cryptographically locked using SHA-256 digests:

| Checkpoint File | Seed | Parameters | Checkpoint SHA-256 Digest |
| :--- | :---: | :---: | :--- |
| `v4_metric_transformer_seed_7.pt` | 7 | 64,833 | `3355e4f1a118c2e4c44f647bf3bcae43444a8b7e41498216c39abaf67659aba6` |
| `v4_metric_transformer_seed_17.pt` | 17 | 64,833 | `81a774c9794a99e6651808d603ba944d4036c0b0faabd6025b702be8f4c36213` |
| `v4_metric_transformer_seed_37.pt` | 37 | 64,833 | `92c6422d14fee7210ef496d00d8136299034f080b2e31904a619c3c658970d40` |

### Architectural Invariants:
- **Input Dimension:** 23 standardized technical and liquidity features.
- **Context Length:** 252 trading sessions, divided into $N_p = 42$ non-overlapping patches of $P = 6$ sessions.
- **Backbone:** 2-layer Transformer Encoder ($d_{\text{model}} = 64$, 4 attention heads, $d_{\text{ff}} = 128$, GELU activations, post-LN).
- **Pooling & Latent Space:** Softmax attention-weighted temporal pooling yielding a 128-dimensional LayerNormed metric embedding $\mathbf{z} \in \mathbb{R}^{128}$ on the unit hypersphere ($||\mathbf{z}||_2 = 1$).

---

## 3. Causal Memory Bank & Retrieval Specification

- **Total Historical Regimes:** 164,871 mature episodic regimes spanning 6 global equity markets (US, India, China, Brazil, France, UK).
- **Temporal Causal Boundary:** Strictly $t_{\text{outcome}} \le \text{2020-12-31}$. No episode maturing after December 31, 2020 exists in the memory bank.
- **Forward Horizon:** 126-session forward holding return ($\text{ret}_{126}$) and forward maximum drawdown ($\text{dd}_{126}$) recorded at maturity.
- **Domestic Universe Isolation Guardrail:** Inquiries for candidate ticker $i$ in market $m$ are strictly partitioned to match historical precedents within market $m$ ($m_{\text{nbr}} = m$) and exclude self-matches ($i_{\text{nbr}} \ne i$).
- **Retrieval Metric:** Cosine similarity $s_{ij} = \mathbf{z}_i^\top \mathbf{z}_j$ in $\mathcal{Z}$.
- **Neighbourhood Size:** Top $K = 25$ nearest historical regimes.
- **Kernel Smoothing:** Nadaraya-Watson kernel with bandwidth $\tau = 0.08$:
  $$w_j = \frac{\exp((s_j - 1)/\tau)}{\sum_{k=1}^K \exp((s_k - 1)/\tau)}$$
- **Expected Return & Tail Risk Estimators:**
  $$\hat{\mu}_i = \sum_{j=1}^{25} w_j \cdot r_j, \quad \widehat{\text{CVaR}}_{0.05, i} = \frac{1}{|T_{0.05}|} \sum_{j \in T_{0.05}} r_j$$
  where $T_{0.05} = \{j : r_j \le \text{VaR}_{0.05}\}$.

---

## 4. Execution Protocol & Trading Invariants

1. **Signal Generation:** Evaluated at session $t$ Close based solely on information available through session $t$.
2. **Execution Timing:** All entries and rebalancings execute at session $t+1$ Open. Zero intraday lookahead.
3. **Transaction Costs:** 10 basis points (0.0010) flat fee deducted from notional capital on all entries, exits, and terminal liquidations.
4. **Scoring Function ($P_0^*$):**
   $$S_i(t) = \frac{\hat{y}_i + 0.8\hat{\mu}_i - 0.2|\widehat{\text{CVaR}}_{0.05, i}|}{v_{21, i} + 10^{-4}}$$
   where $\hat{y}_i$ is the parametric outcome prediction and $v_{21, i}$ is 21-day realized volatility.
5. **Portfolio Selection:** Top $K_{\text{port}} = 3$ non-held candidate assets selected into equal-weighted tranches ($33.3\%$ target capacity).
6. **Risk-Managed Exit (Chandelier Stop):**
   - Dynamic stop distance: $\text{Stop}_t = \max(0.10, 2.5 \times \text{ATR}_{14, t})$.
   - Exit triggered if trailing drawdown from peak price exceeds $\text{Stop}_t$ with minimum holding period $H_{\min} = 1$ session.
   - Maximum holding period: $H_{\max} = 63$ trading sessions.

---

## 5. Pre-Specified Non-Inferiority & Equivalence Criteria

### Primary Hypothesis Test:
We test whether the interpretable retrieval policy $P_0^*$ is non-inferior to the uninterpretable parametric benchmark $P_1$:
$$H_{0, \text{NI}}: \Delta \text{Sharpe} \le -\delta_{\text{tol}} \quad \text{vs.} \quad H_{1, \text{NI}}: \Delta \text{Sharpe} > -\delta_{\text{tol}}$$
where $\Delta \text{Sharpe} = \text{Sharpe}(P_0^*) - \text{Sharpe}(P_1)$.

### Economically Justified Tolerance Margin ($\delta_{\text{tol}}$):
- **Primary Pre-Specified Margin:** $\delta_{\text{tol}} = 0.15$ annualized Sharpe ratio.
- **Economic Justification:** In institutional asset management and regulatory governance (e.g., Basel Committee SR 11-7, MiFID II RTS 6, EU AI Act high-risk classification), black-box neural networks present unquantified epistemic risks and opacity liabilities. A tracking variance up to $0.15$ Sharpe represents the acceptable cost of structural interpretability, provenance tracking, and regulatory auditability.
- **Equivalence Testing (TOST):** We simultaneously evaluate Two One-Sided Tests across the symmetric interval $[-\delta_{\text{tol}}, +\delta_{\text{tol}}]$:
  - Lower bound test: $H_{01}: \Delta \text{Sharpe} \le -0.15$
  - Upper bound test: $H_{02}: \Delta \text{Sharpe} \ge +0.15$
  - Equivalence rejected if $p_{\text{TOST}} = \max(p_{\text{lower}}, p_{\text{upper}}) < 0.05$.

---

## 6. Execution Verification Invariants

Upon the close of trading on **December 31, 2027**, the evaluation pipeline will execute automatically against the prospective 2027 market tape without modification. Any deviation in model code, hyperparameters, or universe definitions constitutes an immediate violation of this registration.
