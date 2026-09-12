# Reproduction Guide: Core-RL-Agent & Memory-Centric Equity Selection

This document provides exact instructions to verify and reproduce the experimental findings and statistical analyses reported in:

> **"A Reproducibility and Robustness Audit of Historical Market Memory for Equity Selection: Negative Validation Across Six Markets"**  
> *Digital Finance* (Springer Nature)  
> Code Repository: [https://github.com/Rohil72/Core-RL-Agent](https://github.com/Rohil72/Core-RL-Agent)  
> Data Repository: [https://github.com/Rohil72/historical-memory-equity-data](https://github.com/Rohil72/historical-memory-equity-data)

---

## 1. Reproduction Levels

We document three distinct levels of reproduction:

### Level 1: Analysis & Statistical Inference from Daily Returns (< 15 seconds)
* **Scope:** Recomputes all 150 run metrics, the multi-market estimand $T(A)$, the mandatory contrast identity $\Delta_{A,B} = T(A) - T(B)$, 10,000 paired block bootstrap draws, and step-down Holm-Bonferroni corrections from unrounded daily portfolio returns.
* **Requirements:** Python 3.9+, `numpy`, `pandas`, `pyarrow` (CPU-only, no GPU required).
* **Command:**
  ```bash
  python -m memory_study.reproduce_release --bundle ../historical-memory-equity-data --output ./replay_output
  python -m memory_study.validate_release --bundle ../historical-memory-equity-data --replay ./replay_output
  ```
* **Expected Output:** All primary and secondary contrast point estimates match the published tables to $\le 10^{-10}$ floating-point precision.

### Level 2: Portfolio Simulation from Checkpoints (< 5 minutes)
* **Scope:** Replays the institutional execution state machine (3 slots, Chandelier exit, 63 sessions max hold, 10 bps transaction fees) across continuous 2024 portfolios using precomputed feature representations and trained gate checkpoints.
* **Requirements:** PyTorch, CUDA recommended (CPU compatible).
* **Command:**
  ```bash
  python -m memory_study.run_final_comparison
  ```

### Level 3: End-to-End Retraining from Raw Market Data
* **Scope:** Retrains the Patch Transformer and 2-layer MLP encoders on 2013–2020 history from raw Yahoo Finance price series, fits market scalers, and fits the selective trust gates.
* **Requirements:** NVIDIA GPU with $\ge 4$GB VRAM, CUDA 12.1+, ~30 minutes runtime.

---

## 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Rohil72/Core-RL-Agent.git
cd Core-RL-Agent
git checkout paper-v1.0.0

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
pytest tests/test_release_contract.py tests/test_final_analysis_reconciliation.py -v
```

All tests must pass with exit code 0.
