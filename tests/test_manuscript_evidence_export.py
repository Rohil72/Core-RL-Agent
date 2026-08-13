import json
import tarfile
from pathlib import Path

from scripts.export_manuscript_evidence import export_bundle


def test_reviewer_bundle_hashes_artifacts_and_reports_missing_groups(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "comparison").mkdir(parents=True)
    (run / "experiment_manifest.yaml").write_text("run: test\n", encoding="utf-8")
    (run / "comparison" / "verdict.json").write_text('{"status": "done"}', encoding="utf-8")
    output = tmp_path / "bundle.tar.gz"

    result = export_bundle([str(run)], str(output), "reviewer")

    assert result["status"] == "completed"
    assert output.is_file()
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
        assert "manuscript_evidence/inventory.json" in names
        assert "manuscript_evidence/missing_evidence.json" in names
        inventory = json.load(archive.extractfile("manuscript_evidence/inventory.json"))
        assert any(row["archive_path"].endswith("experiment_manifest.yaml") for row in inventory)
        missing = json.load(archive.extractfile("manuscript_evidence/missing_evidence.json"))
        assert any(row["group"] == "trade_ledgers" for row in missing["missing_artifact_groups"])


def test_full_profile_includes_latents_while_reviewer_profile_does_not(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "latents").mkdir(parents=True)
    (run / "experiment_manifest.yaml").write_text("run: test\n", encoding="utf-8")
    (run / "latents" / "test_latents.parquet").write_bytes(b"latent")

    reviewer = tmp_path / "reviewer.tar.gz"
    full = tmp_path / "full.tar.gz"
    export_bundle([str(run)], str(reviewer), "reviewer")
    export_bundle([str(run)], str(full), "full")

    with tarfile.open(reviewer, "r:gz") as archive:
        assert not any(name.endswith("test_latents.parquet") for name in archive.getnames())
    with tarfile.open(full, "r:gz") as archive:
        assert any(name.endswith("test_latents.parquet") for name in archive.getnames())
