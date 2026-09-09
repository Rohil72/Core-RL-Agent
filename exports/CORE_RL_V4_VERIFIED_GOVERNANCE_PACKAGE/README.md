# Core-RL V4: Verified Governance and Empirical Evaluation Package
===================================================================
Package: CORE_RL_V4_VERIFIED_GOVERNANCE_PACKAGE
Date: September 9, 2026
Author: Core-RL Research Team

## Overview & Scientific Dual Identity
This package contains the authoritative empirical evidence, pre-trained model weights, synchronized publication tables, and verification tooling for the Core-RL framework.

The framework is organized around two complementary identities:
1. **The Long-Horizon Alpha Backbone (Patch Temporal Transformer):**
   - Ingests $T = 252$ trading days aggregated into $N_p = 42$ macro patches ($P = 6$ days).
   - Learns continuous metric geometry on the unit hypersphere $\mathbb{S}^{127}$, achieving rank correlation $\rho = +0.486$.
   - Delivers primary predictive utility (+3.54% return, 0.271 Sharpe, Sortino 0.356).
2. **The Regulatory Governance Layer (Retrospective Case-Based Memory):**
   - Retrospective memory bank ($N = 164,871$ mature regimes $\le 2020$).
   - Provides point-in-time precedent retrieval and strict audit trails (SR 11-7, MiFID II RTS 6, EU AI Act).
   - Provides Expected Shortfall tail-risk defense ($\text{CVaR}_{0.05}$, $\tau = 0.08$).
   - Establishes statistical non-inferiority via 10,000-draw multi-block bootstrap TOST ($\Delta\text{Sharpe} = +0.0108$, $p = 0.0455 < 0.05$ under $\delta = 0.15$).

---

## Directory Structure
- `models/`: 3 frozen PyTorch model checkpoints (Seeds 7, 17, 37).
- `evidence/`: 18 comprehensive empirical evidence and authoritative runtime truth files.
- `latex_tables/`: 9 publication-ready LaTeX tables formatted with booktabs.
- `VERIFY_PACKAGE.py`: Counter-adversarial verification script.
- `SHA256SUMS.txt`: Cryptographic manifest.

---

## One-Click Reproduction & Verification
To verify the entire package against cryptographic checksums, timestamp causality, top-5 neighbor agreement, candidate scores, and table alignment, execute:
```bash
python VERIFY_PACKAGE.py
```
