# Prospective Evaluation Protocol & Formal Architecture Freeze
============================================================
Registration Date: September 9, 2026
Prospective Evaluation Horizon: Calendar Year 2027 (2027-01-01 to 2027-12-31)
Package: CORE_RL_V4_VERIFIED_GOVERNANCE_PACKAGE

## 1. Executive Summary & Regulatory Purpose
To resolve referee concerns regarding potential data snooping and post-hoc hyperparameter tweaking, this protocol establishes a prospective evaluation registration for Core-RL V4. All model architectures, hyperparameters, feature extraction logic, scalers, and the non-inferiority margin are permanently frozen as of September 9, 2026, prior to the unrolling of the 2027 evaluation period.

## 2. Frozen Artifacts & Checksum Lineage
The evaluation will be conducted strictly using the frozen weights in `models/`:
- `v4_metric_transformer_seed_7.pt` (SHA-256: `3355e4f1a118c2e4c44f647bf3bcae43444a8b7e41498216c39abaf67659aba6`, 77,250 parameters)
- `v4_metric_transformer_seed_17.pt` (SHA-256: `81a774c9794a99e6651808d603ba944d4036c0b0faabd6025b702be8f4c36213`, 77,250 parameters)
- `v4_metric_transformer_seed_37.pt` (SHA-256: `92c6422d14fee7210ef496d00d8136299034f080b2e31904a619c3c658970d40`, 77,250 parameters)

## 3. Retrospective Memory Bank Registration
- Total Regimes: 164,871 mature market episodes.
- Strict Temporal Cutoff: Precedents must strictly satisfy date $\le 2020-12-31$.
- Scope: Domestic equity universe guardrail ($P_0^*$).

## 4. Statistically Pre-Specified Equivalence Hypotheses
- Primary Hypothesis: Two One-Sided Tests (TOST) non-inferiority of $P_0^*$ vs $P_1$.
- Non-inferiority margin: $\delta_{\text{tol}} = 0.15$ annualized Sharpe.
- Primary block length: $L = 21$ trading sessions.
- Significance level: $\alpha = 0.05$.
