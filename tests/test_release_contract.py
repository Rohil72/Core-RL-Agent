"""
Unit tests for release contract and evidence verification.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_release_contract_files_exist():
    contract_path = PROJECT_ROOT / "release" / "experiment_contract.json"
    env_path = PROJECT_ROOT / "release" / "environment.json"
    prov_path = PROJECT_ROOT / "release" / "provenance.json"
    license_path = PROJECT_ROOT / "LICENSE"
    
    assert contract_path.exists(), "experiment_contract.json missing"
    assert env_path.exists(), "environment.json missing"
    assert prov_path.exists(), "provenance.json missing"
    assert license_path.exists(), "LICENSE missing"
    
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["models"]["mlp_backbone"]["latent_dimension"] == 128
    assert contract["retrieval"]["nearest_neighbors_k"] == 25
    assert contract["statistical_inference"]["bootstrap_draws"] == 10000

def test_reanalysis_artifacts_integrity():
    reanalysis_dir = PROJECT_ROOT / "research_runs" / "memory_study" / "final_comparison" / "reanalysis_v1"
    
    master_df = pd.read_csv(reanalysis_dir / "master_performance.csv")
    assert len(master_df) == 11, f"Expected 11 arms in master_performance.csv, found {len(master_df)}"
    
    primary_df = pd.read_csv(reanalysis_dir / "primary_contrasts.csv")
    assert len(primary_df) == 5, f"Expected 5 primary contrasts, found {len(primary_df)}"
    
    # Check identity on all rows
    for row in primary_df.itertuples():
        diff = row.candidate_metric - row.comparator_metric
        assert abs(diff - row.delta_original) <= 1.05e-4, f"Identity violation for {row.contrast_id}"
