# Analytical Survivorship Bias Bounding and Universe Boundary Contract

**Target Journal:** Digital Finance (Springer Nature)  
**Execution Timestamp:** September 6, 2026  
**Universe Scope:** Manually configured 103 liquid large-cap securities across 6 global equity markets.

---

## 1. Explicit Boundary Disclosure
As disclosed in the manuscript, the evaluated 103-security universe was manually selected based on historical prominence and trading liquidity rather than formed as a dynamic, point-in-time constituent panel with automated delisting capture. 

**Boundary Restriction:** All empirical findings in this study are strictly valid within the configured 103-security universe and **cannot be generalized to the broader cross-section of equities, small-cap stocks, or illiquid markets**.

---

## 2. Quantitative Survivorship Bias Bound

Let:
- $\lambda_{\text{delist}}$ denote the annual delisting rate for large-cap equities within major developed and emerging indices (empirically $1.0\% - 1.5\%$ annually; Shumway, 1997; Jensen et al., 2023).
- $R_{\text{survivor}}$ denote the annual return of the surviving large-cap cross-section.
- $R_{\text{delist}}$ denote the average terminal realized return of delisted constituents (typically experiencing an average terminal distress shock of $-30\%$ to $-50\%$; Shumway, 1997).

The maximum expected upward return distortion in the observed portfolio return $R_{\text{obs}}$ due to survivorship conditioning is analytically bounded by:
$$\Delta R_{\text{survivorship}} \le \lambda_{\text{delist}} \times \left( R_{\text{survivor}} - R_{\text{delist}} \right)$$

Under conservative empirical parameters:
- $\lambda_{\text{delist}} = 0.015$ ($1.5\%$ annual failure/delisting rate for prominent index constituents)
- Distress drop: $R_{\text{survivor}} - R_{\text{delist}} \le 0.06 - (-0.40) = 0.46$ ($46\%$ return differential)

$$\Delta R_{\text{survivorship}} \le 0.015 \times 0.46 = 0.0069 \quad (69\text{ basis points})$$

### Empirical Implications:
1. **Identical Cross-System Impact:** All seven evaluated systems ($P_0, P_0^*, P_1, P_2, P_3, P_4, P_6$) operate on the exact identical 103-security universe. Therefore, the **paired differences** ($\Delta \text{Sharpe}(P_0 - P_1)$, $\Delta \text{Sharpe}(P_0 - P_3)$) cancel out first-order universe-selection bias.
2. **Negative Result Remains Robust:** Because the principal finding is a **neutral/negative result** ($P_0$ underperforms $P_1$ by $-10.7$ bps Sharpe, $P_0$ underperforms passive $P_6$ by $-34.6$ bps Sharpe), survivorship bias would, if anything, have artificially inflated performance. The fact that external memory fails to outperform simple controls even within a prominent survivor basket strengthens, rather than weakens, the negative empirical finding.
