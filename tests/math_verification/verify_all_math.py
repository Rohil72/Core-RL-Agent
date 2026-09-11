#!/usr/bin/env python3
"""
Formal Mathematical Verification Runner & Audit Certificate Generator
======================================================================
Core-RL Research Mathematical Verification Suite

Runs symbolic derivations, axiomatic metric proofs, statistical guarantees,
and numerical invariant checks across all mathematical domains in the research:

1. Candidate Scoring & Decision Allocation Math
2. Metric Geometry & Kernel Weight Aggregation
3. Loss Functions, Outcome Geometry & Representation Math
4. Kaplan-Meier Survival Analysis & Greenwood Uncertainty
5. Portfolio Execution, Slippage & Equity Math
6. Statistical Metrics, Bootstrap & Multiplicity Control
7. Feature Engineering & Z-Score Scaling Math
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import pytest

def main() -> int:
    start_time = time.time()
    print("=" * 80)
    print("  CORE-RL FORMAL MATHEMATICAL VERIFICATION SUITE")
    print("=" * 80)
    print("Timestamp: ", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    print("Python:    ", sys.version.split()[0])
    print("NumPy:     ", np.__version__)
    print()

    verification_dir = Path(__file__).parent
    test_files = sorted(verification_dir.glob("test_*.py"))

    print(f"[*] Discovered {len(test_files)} mathematical verification modules:")
    for f in test_files:
        print(f"    - {f.name}")
    print()

    # Run pytest programmatically on the verification suite
    args = [
        str(verification_dir),
        "-v",
        "--tb=short",
        "-ra",
    ]

    print("[*] Executing machine verification test suite...")
    exit_code = pytest.main(args)
    elapsed = time.time() - start_time

    print()
    print("=" * 80)
    if exit_code == 0:
        print(f"  [PASSED] ALL MATHEMATICAL INVARIANTS VERIFIED IN {elapsed:.2f}s")
        print("  Mathematical certificate: 100% of analytical, symbolic, and property")
        print("  checks across the Core-RL research have passed machine verification.")
    else:
        print(f"  [FAILED] Verification failed with exit code {exit_code}")
    print("=" * 80)

    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
