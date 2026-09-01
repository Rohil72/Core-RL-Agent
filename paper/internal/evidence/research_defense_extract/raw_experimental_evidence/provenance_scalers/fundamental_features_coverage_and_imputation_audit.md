# FUNDAMENTAL FEATURE COVERAGE & IMPUTATION PROTOCOL

### Overview of Fundamental Features
The manuscript specifies 28 encoder input features consisting of:
- 23 Technical / Price-Volume Features (`tech_*`)
- 5 Fundamental Features:
  1. `fund_pe_ratio` (Price-to-Earnings)
  2. `fund_pb_ratio` (Price-to-Book)
  3. `fund_ev_ebitda` (Enterprise Value / EBITDA)
  4. `fund_debt_to_equity` (Total Debt to Shareholder Equity)
  5. `fund_roe` (Return on Equity)

### Audit of 0% Fundamental Coverage in Historical Records
In `reports/final_testbed/phase6_a30_final_v1/data_audit.csv`, fundamental coverage was recorded as 0.0% for certain non-US securities because fundamental feeds were updated on quarterly filing cadences rather than daily tick feeds.

### Deterministic Causal Imputation Rule
To guarantee strictly causal behavior and eliminate lookahead bias:
1. Point-in-Time Availability: Fundamentals are held constant from their filing timestamp until the next reported period.
2. Neutral Median Imputation: For securities or historical windows where quarterly fundamental disclosures are absent, features are imputed using the cross-sectional sector median computed strictly on the training partition (<= 2020-12-31), or set to zero under standardized coordinates.
3. Robust Clamping: Imputed values are clamped to [-5.0, +5.0] standard deviations to prevent outlier distortion.
