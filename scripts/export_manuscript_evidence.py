"""Create a hashed, reviewer-facing archive from one or more experiment runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REVIEWER_PATTERNS = (
    "experiment_manifest.yaml",
    "build_summary.json",
    "**/contract.json",
    "**/migrations/*.json",
    "generated_configs/**/*.yaml",
    "comparison/**/*",
    "selection/**/*",
    "**/metrics.json",
    "**/trades.csv",
    "**/signals.parquet",
    "**/neighbors.parquet",
    "**/training_complete.json",
    "**/training_summary.json",
    "**/model_audit.json",
    "**/source_manifest.json",
    "**/market_audit*.csv",
    "**/exclusion*.csv",
    "**/reliability*.csv",
    "**/paired_results.csv",
    "**/variant_summary.csv",
    "**/verdict.json",
    "**/provenance*.json",
)

FULL_PATTERNS = REVIEWER_PATTERNS + (
    "latents/**/*.parquet",
    "decisions/**/*.parquet",
    "models/**/*.pt",
    "adapters/**/*.pt",
    "orchestration_state/logs/*.log",
    "orchestration_state/jobs/*.json",
)

EVIDENCE_GROUPS = {
    "experiment_manifest": ("experiment_manifest.yaml",),
    "resolved_configuration": ("generated_configs/**/*.yaml",),
    "durable_contract": ("**/contract.json",),
    "comparison_outputs": ("comparison/**/*", "selection/**/*"),
    "per_run_metrics": ("**/metrics.json",),
    "trade_ledgers": ("**/trades.csv",),
    "retrieval_signals": ("**/signals.parquet",),
    "neighbor_ledgers": ("**/neighbors.parquet",),
    "training_completion": ("**/training_complete.json", "**/training_summary.json"),
}

EXTERNAL_LIMITATIONS = (
    "Point-in-time historical constituent membership is not established by experiment outputs alone.",
    "Dividend, corporate-action, FX, calendar-alignment, and capital-pooling treatment require an executed data/backtest contract.",
    "A fixed manually selected liquid-company universe retains survivorship and researcher-selection bias.",
    "An already inspected temporal interval cannot become a fresh untouched confirmation set by rerunning it.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": command, "error": str(exc)}


def _resolve_roots(values: Iterable[str]) -> list[Path]:
    roots: list[Path] = []
    for value in values:
        path = Path(value)
        resolved = (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Experiment run root does not exist: {resolved}")
        roots.append(resolved)
    return roots


def _collect(roots: list[Path], patterns: tuple[str, ...]) -> list[Path]:
    files: set[Path] = set()
    for root in roots:
        for pattern in patterns:
            files.update(path.resolve() for path in root.glob(pattern) if path.is_file())
    return sorted(files, key=lambda path: str(path).lower())


def _archive_label(path: Path) -> str:
    try:
        return (Path("project") / path.relative_to(PROJECT_ROOT)).as_posix()
    except ValueError:
        return (Path("external") / path.parent.name / path.name).as_posix()


def _evidence_status(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, patterns in EVIDENCE_GROUPS.items():
        matches = _collect(roots, patterns)
        rows.append(
            {
                "group": name,
                "status": "present" if matches else "missing",
                "file_count": len(matches),
                "examples": [_archive_label(path) for path in matches[:5]],
            }
        )
    return rows


def _provenance(profile: str, roots: list[Path]) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "project_root": str(PROJECT_ROOT),
        "run_roots": [str(path) for path in roots],
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "git_head": _run(["git", "rev-parse", "HEAD"]),
        "git_status": _run(["git", "status", "--short"]),
        "git_remote": _run(["git", "remote", "-v"]),
        "python_packages": _run([sys.executable, "-m", "pip", "freeze"]),
        "gpu": _run(["nvidia-smi", "-q"]),
    }


def export_bundle(
    run_roots: Iterable[str],
    output: str,
    profile: str = "reviewer",
) -> dict[str, Any]:
    """Export selected evidence, checksums, provenance, and a missing-data ledger."""
    roots = _resolve_roots(run_roots)
    output_path = Path(output)
    output_path = (PROJECT_ROOT / output_path).resolve() if not output_path.is_absolute() else output_path.resolve()
    if output_path.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError("Output must end in .tar.gz")
    patterns = FULL_PATTERNS if profile == "full" else REVIEWER_PATTERNS
    files = [path for path in _collect(roots, patterns) if path != output_path]
    evidence_status = _evidence_status(roots)

    inventory = [
        {
            "source": str(path),
            "archive_path": _archive_label(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    missing = {
        "missing_artifact_groups": [row for row in evidence_status if row["status"] == "missing"],
        "external_methodological_limitations": list(EXTERNAL_LIMITATIONS),
        "note": "Missing means absent from the supplied run roots, not necessarily absent from every historical VM run.",
    }
    provenance = _provenance(profile, roots)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="manuscript-evidence-") as temporary:
        metadata = Path(temporary)
        (metadata / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        (metadata / "evidence_status.json").write_text(json.dumps(evidence_status, indent=2), encoding="utf-8")
        (metadata / "missing_evidence.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")
        (metadata / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        (metadata / "checksums.sha256").write_text(
            "\n".join(f"{row['sha256']}  {row['archive_path']}" for row in inventory) + "\n",
            encoding="utf-8",
        )
        (metadata / "README.md").write_text(
            "# Manuscript Evidence Bundle\n\n"
            f"Profile: `{profile}`  \nFiles: `{len(inventory)}`  \n"
            f"Payload bytes: `{sum(row['size_bytes'] for row in inventory)}`\n\n"
            "`inventory.json` and `checksums.sha256` authenticate included artifacts. "
            "`evidence_status.json` records claim-supporting artifact groups. "
            "`missing_evidence.json` distinguishes absent files from methodological limitations.\n",
            encoding="utf-8",
        )
        with tarfile.open(output_path, "w:gz") as archive:
            for name in (
                "README.md",
                "inventory.json",
                "checksums.sha256",
                "evidence_status.json",
                "missing_evidence.json",
                "provenance.json",
            ):
                archive.add(metadata / name, arcname=f"manuscript_evidence/{name}")
            for row, path in zip(inventory, files):
                archive.add(path, arcname=f"manuscript_evidence/{row['archive_path']}")

    result = {
        "status": "completed",
        "profile": profile,
        "output": str(output_path),
        "file_count": len(files),
        "payload_bytes": sum(row["size_bytes"] for row in inventory),
        "archive_bytes": output_path.stat().st_size,
        "archive_sha256": _sha256(output_path),
        "missing_artifact_groups": [row["group"] for row in evidence_status if row["status"] == "missing"],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("reviewer", "full"), default="reviewer")
    args = parser.parse_args()
    print(json.dumps(export_bundle(args.run_root, args.output, args.profile), indent=2))


if __name__ == "__main__":
    main()
