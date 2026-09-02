# Audit Defense Clarification 02: External Evaluation Protocol & India Dataset Boundary

## 1. Findings / Criticisms Addressed
> *"The 2025 'P0' evaluation does not actually apply memory/model ranking."*  
> *"India remains frozen after March 31, 2025."*

## 2. Explanation of the India Cutoff Date
* **Origin of Boundary:** The frozen historical data snapshot extracted from the research environment contained National Stock Exchange of India (NSE) price series concluding on **31 March 2025**. Western and East Asian exchanges in the dataset extended through December 2025.
* **Why the Series Was Frozen (Not Synthesized):**
  * Rather than generating synthetic or imputed prices for Indian equities after 31 March 2025 (which would violate econometric auditing standards and introduce hallucinated market returns), the portfolio equity for India was deliberately **frozen at its last valid trading session**.
  * This is standard quantitative practice in empirical finance when evaluating multi-market portfolios across non-synchronous dataset boundaries.
  * In the manuscript, this is transparently disclosed as a **dataset boundary condition**, not an active trading decision.

## 3. Rectification of the 2025 P0 Policy Execution
* **Previous Implementation:** An earlier evaluation script compiled an equal-weighted benchmark of available market constituent returns rather than driving capital allocations via model signals.
* **Corrected P0 Execution Contract:**
  1. **Candidate Screening:** The Global Temporal Transformer encoder and causal memory bank generate embeddings and retrieve historical analogues for all available market equities on day $T$.
  2. **Deterministic Ranking:** The P0 opportunity score (integrating upside, downside, path quality, and uncertainty) ranks candidates.
  3. **Order Execution:** Capital enters top-ranked candidates at the next session open ($T+1$) with a mandatory 10 bps transaction fee.
  4. **Risk & Holding Controls:** Positions are subject to a 5-session minimum holding restriction, a 63-session maximum holding window, and a 10% trailing stop-loss.
* **Resulting Finding:** When true P0 execution is applied out-of-sample in 2025, the policy fails to outperform passive buy-and-hold benchmarks, reinforcing the central **negative validation** thesis of the paper.
