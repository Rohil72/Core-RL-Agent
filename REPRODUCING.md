# Reproduction Guide: Core-RL-Agent & Memory-Centric Equity Selection

This document provides exact instructions to verify and reproduce the experimental findings and statistical analyses reported in:

> **"Auditable Historical-Memory Retrieval for Long-Horizon Equity Selection: Negative Validation Across Six Sovereign Markets"
> Authors: Rohil Gujarathi, Sangeeta Oswal, Vaibhav Goyal**  
> *Digital Finance* (Springer Nature)  
> Code Repository: [https://github.com/Rohil72/Core-RL-Agent](https://github.com/Rohil72/Core-RL-Agent)  
> Data Repository: [https://github.com/Rohil72/historical-memory-equity-data](https://github.com/Rohil72/historical-memory-equity-data)

---

## 1. Reproduction Levels & Verified Execution Scope

We distinguish three levels of reproduction, clarifying what is independently recomputed versus carried forward:

### Level 1: Analysis & Statistical Inference from Daily Returns (< 15 seconds)
* **Status:** **Demonstrated & Verified Public Reproduction.**
* **Recomputed Metrics:** Independently recalculates annualized returns, annual volatilities, Sharpe ratios, primary C1–C5 and secondary contrast point estimates ($\Delta_{A,B} = T(A) - T(B)$), 95% bootstrap confidence intervals, empirical $p$-values, step-down Holm-Bonferroni family corrections, block length sensitivity across $L \in \{2, 4, 8\}$ weeks, and 10,000-draw bootstrap distributions directly from unrounded daily portfolio returns.
* **Carried-Forward Reference Metrics:** Max Drawdown, Daily Win Rate, Portfolio Turnover, Active Exposure, Prediction MSE, and Rank IC are carried forward from `data/master_performance.csv` for display completeness; they are not independently re-simulated from order fills in Level 1.
* **Statistical Bootstrap Specification:** First-order centered calendar-aligned weekly block bootstrap (non-wrapping calendar-week blocks of 2, 4, and 8 weeks; 10,000 draws).
* **Requirements:** Python 3.9+, `numpy`, `pandas`, `pyarrow` (CPU-only, no GPU required).
* **Commands:**
  ```bash
  # Deterministic replay from public bundle data
  python -m memory_study.reproduce_release --bundle ../historical-memory-equity-data --output ../historical-memory-equity-data/validation/replay_output

  # Full two-tier release validation (bundle consistency + replay verification)
  python -m memory_study.validate_release --bundle ../historical-memory-equity-data --replay ../historical-memory-equity-data/validation/replay_output
  ```
* **Expected Output:** All primary and secondary contrast point estimates, confidence intervals, p-values, sensitivity tables, and bootstrap draws match reference outputs to $\le 10^{-10}$ floating-point precision.

### Level 2: Portfolio Simulation from Model Checkpoints (< 5 minutes)
* **Status:** **Code & Checkpoints Provided; Intermediate Caches Excluded.**
* **Scope:** Replays the institutional execution state machine (3 slots, Chandelier exit, 63 sessions max hold, 10 bps transaction fees) across continuous 2024 portfolios using precomputed feature representations and trained gate checkpoints.
* **Availability Note:** Intermediate feature and embedding cache tensors (~several GB) are excluded from the public data bundle to keep the download compact (< 4 MB). Executing Level 2 re-simulation requires downloading the intermediate cache files (available upon request) or generating them via Level 3.
* **Requirements:** PyTorch, CUDA recommended (CPU compatible).

### Level 3: End-to-End Retraining from Raw Market Data
* **Status:** **Code Provided; Subject to External Data Provider APIs.**
* **Scope:** Retrains the Patch Transformer and MLP encoders on historical price-volume data, fits market scalers, and fits selective trust gates.
* **Requirements:** External market data vendor API access, NVIDIA GPU with $\ge 4$GB VRAM.

---

## 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Rohil72/Core-RL-Agent.git
cd Core-RL-Agent
git checkout paper-v1.0.2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Automated Validation Test Suite

Run the full pytest suite to verify all causal invariants, mathematical identities, and release contracts:

```bash
pytest tests/test_release_contract.py tests/test_validation_suite.py tests/test_final_analysis_reconciliation.py -v
```

All tests must pass with exit code 0.
