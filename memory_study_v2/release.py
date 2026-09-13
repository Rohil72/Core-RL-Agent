"""Release packaging, checksum generation, and reproduction verification (v2).

Acceptance criteria addressed:
- Full-precision summaries, primary contrasts, daily returns, and transaction ledgers.
- Comprehensive checksum manifest generation (release_checksums.json).
- Standalone verification and replay validation (R10, R12).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from memory_study_v2.artifacts import sha256_file, to_canonical_json
from memory_study_v2.inference import replay_analysis_bundle


def generate_release_checksums(release_dir: Union[str, Path]) -> Dict[str, str]:
    """Compute SHA-256 checksums of all release artifacts."""
    p = Path(release_dir)
    checksums: Dict[str, str] = {}
    for f in sorted(p.rglob("*")):
        if f.is_file() and f.name != "release_checksums.json" and not f.name.endswith(".checksums.json"):
            rel_name = str(f.relative_to(p)).replace("\\", "/")
            checksums[rel_name] = sha256_file(f)

    manifest_path = p / "release_checksums.json"
    with open(manifest_path, "w", encoding="utf-8") as out:
        out.write(to_canonical_json(checksums))

    return checksums


def verify_release_bundle(release_dir: Union[str, Path]) -> Dict[str, Any]:
    """Verify integrity of all files in release bundle and validate statistical replay."""
    p = Path(release_dir)
    manifest_path = p / "release_checksums.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing checksums manifest at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        declared_checksums = json.load(f)

    mismatches = []
    for rel_path, declared_hash in declared_checksums.items():
        actual_file = p / rel_path
        if not actual_file.exists():
            mismatches.append(f"Missing file: {rel_path}")
            continue
        actual_hash = sha256_file(actual_file)
        if actual_hash != declared_hash:
            mismatches.append(f"Hash mismatch for {rel_path}: declared {declared_hash}, actual {actual_hash}")

    if mismatches:
        raise ValueError(f"Release verification failed with errors: {mismatches}")

    # Replay analysis bundle
    replay_status = replay_analysis_bundle(p)

    return {
        "status": "RELEASE_VERIFIED",
        "verified_files_count": len(declared_checksums),
        "replay_status": replay_status,
    }
