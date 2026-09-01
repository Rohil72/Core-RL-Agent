"""Unit and Integration Tests for Research Defense & Paper Evidence Extractor."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.eval.research_defense_extractor import (
    CausalityArtifactBuilder,
    DefensePackageBuilder,
    LatexTableGenerator,
    MarkdownReportGenerator,
    ProvenanceCollector,
    export_research_defense_bundle,
)


def test_provenance_collector() -> None:
    prov = ProvenanceCollector.collect()
    assert "host" in prov
    assert "platform" in prov["host"]
    assert "python_version" in prov["host"]
    assert "gpu" in prov
    assert "git" in prov
    assert "python_packages_lock" in prov


def test_causality_artifact_builder() -> None:
    data = CausalityArtifactBuilder.build_all()
    assert "target_lineage" in data
    assert len(data["target_lineage"]) == 11
    assert "target_252_isolation" in data
    assert data["target_252_isolation"]["status"] == "PASS"
    assert data["target_252_isolation"]["violations"] == 0
    assert "universe_retention" in data
    assert data["universe_retention"]["totals"]["requested"] == 108
    assert data["universe_retention"]["totals"]["available"] == 103
    assert data["universe_retention"]["totals"]["excluded"] == 5


def test_latex_table_rendering() -> None:
    causality_data = CausalityArtifactBuilder.build_all()
    
    # Universe table
    uni_tex = LatexTableGenerator.render_universe_retention_table(causality_data["universe_retention"])
    assert r"\begin{table}" in uni_tex
    assert r"\caption{Retained Multi-Market Universe" in uni_tex
    assert r"\label{tab:universe_retention}" in uni_tex
    assert r"\end{table}" in uni_tex

    # Lineage table
    lin_tex = LatexTableGenerator.render_target_lineage_table(causality_data["target_lineage"])
    assert r"\begin{table}" in lin_tex
    assert "future_max_return_252" in lin_tex
    assert r"\end{table}" in lin_tex

    # Primary systems table
    mock_primary_df = pd.DataFrame([
        {"System": "P0", "Name": "Full memory", "Total Return": 0.2261, "Sharpe": 1.084, "Max Drawdown": -0.1716, "Win Rate": 0.70, "Trades": 70},
        {"System": "P1", "Name": "No memory", "Total Return": 0.0410, "Sharpe": 0.210, "Max Drawdown": -0.2240, "Win Rate": 0.485, "Trades": 68},
    ])
    pri_tex = LatexTableGenerator.render_primary_systems_table(mock_primary_df)
    assert r"\begin{table}" in pri_tex
    assert "P0" in pri_tex
    assert "22.61\\%" in pri_tex
    assert r"\end{table}" in pri_tex


def test_markdown_report_rendering() -> None:
    prov = ProvenanceCollector.collect()
    causality_data = CausalityArtifactBuilder.build_all()
    summary = MarkdownReportGenerator.render_executive_summary(
        title="Test Research Defense",
        provenance=prov,
        causality=causality_data,
        eval_summary={},
    )
    assert "# Test Research Defense" in summary
    assert "H1 (State Representation)" in summary
    assert "H2 (Causal Memory Utility)" in summary
    assert "H3 (Distributional Evidence)" in summary

    checklist = MarkdownReportGenerator.render_reproducibility_checklist(prov)
    assert "Reproducibility Checklist" in checklist


def test_defense_package_builder_and_verify_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "experiment_run"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text('{"sharpe": 1.25, "total_return": 0.35}', encoding="utf-8")
    (run_dir / "trades.csv").write_text("ticker,entry_date,exit_date,pnl,net_return\nAAPL,2024-01-02,2024-01-15,150.0,0.05\n", encoding="utf-8")

    bundle_dest = tmp_path / "output_bundle.tar.gz"

    builder = DefensePackageBuilder(
        run_roots=[run_dir],
        output_path=bundle_dest,
        profile="reviewer",
        bundle_title="Unit Test Defense Bundle",
    )

    result = builder.export_archive(export_format="all")
    assert result["status"] == "completed"
    assert result["file_count"] > 10
    assert bundle_dest.exists()

    # Check that the uncompressed directory was also created
    dir_dest = tmp_path / "output_bundle"
    assert dir_dest.is_dir()
    assert (dir_dest / "verify_bundle.py").exists()
    assert (dir_dest / "checksums.sha256").exists()
    assert (dir_dest / "inventory.json").exists()
    assert (dir_dest / "provenance" / "hardware_and_environment.json").exists()
    assert (dir_dest / "causality_and_data_integrity" / "target_lineage_table.csv").exists()
    assert (dir_dest / "causality_and_data_integrity" / "target_252_isolation_proof.json").exists()
    assert (dir_dest / "evaluation_matrices" / "primary_systems_p0_p6.csv").exists()
    assert (dir_dest / "evaluation_matrices" / "statistical_significance_tests.csv").exists()
    assert (dir_dest / "manuscript_tables_latex" / "table_universe_retention.tex").exists()
    assert (dir_dest / "manuscript_reports_markdown" / "RESEARCH_DEFENSE_EXECUTIVE_SUMMARY.md").exists()

    # Test the standalone validator script inside the generated bundle
    verify_script = dir_dest / "verify_bundle.py"
    proc = subprocess.run([sys.executable, str(verify_script)], cwd=dir_dest, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "SUCCESS: ALL ARTIFACTS VERIFIED AND AUTHENTICATED." in proc.stdout


def test_export_convenience_function(tmp_path: Path) -> None:
    output_tar = tmp_path / "quick_defense.tar.gz"
    res = export_research_defense_bundle(
        run_roots=[tmp_path],
        output=output_tar,
        export_format="tar.gz",
    )
    assert res["status"] == "completed"
    assert output_tar.exists()
    with tarfile.open(output_tar, "r:gz") as archive:
        names = archive.getnames()
        assert any("inventory.json" in n for n in names)
        assert any("checksums.sha256" in n for n in names)
        assert any("verify_bundle.py" in n for n in names)
