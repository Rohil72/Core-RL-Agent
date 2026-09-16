"""Unit tests for targeted recovery pipeline."""

import json
from pathlib import Path
import pytest

from scripts.recover_a30_targeted import execute_targeted_recovery, run_fold_2021_equivalence_gate


def test_targeted_recovery_argument_parsing():
    import subprocess
    res = subprocess.run(
        ["python3", "scripts/recover_a30_targeted.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "--gate-only" in res.stdout
    assert "--recovered-dir" in res.stdout
    assert "--source-dir" in res.stdout
