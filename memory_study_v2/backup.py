"""Artifact backup, off-VM retrieval, hash verification, and resume guards (v2).

Acceptance criteria addressed:
- A34: Completed fold and resumable checkpoint retrieved off-VM with matching hashes.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from memory_study_v2.artifacts import (
    ArtifactMetadata,
    CorruptedArtifactError,
    IncompleteArtifactError,
    ProvenanceError,
    read_atomic_artifact,
    sha256_file,
    validate_provenance,
)


def export_artifact_to_backup(
    artifact_path: Union[str, Path],
    backup_dir: Union[str, Path],
) -> Tuple[Path, Path, str]:
    """Export an atomic artifact and its metadata to a backup directory, verifying byte SHA-256."""
    src_payload = Path(artifact_path).resolve()
    src_meta = src_payload.parent / f"{src_payload.name}.meta.json"

    if not src_payload.exists():
        raise FileNotFoundError(f"Source artifact not found: {src_payload}")
    if not src_meta.exists():
        raise IncompleteArtifactError(f"Source metadata missing: {src_meta}")

    dst_dir = Path(backup_dir).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_payload = dst_dir / src_payload.name
    dst_meta = dst_dir / src_meta.name

    # Copy files
    shutil.copy2(src_payload, dst_payload)
    shutil.copy2(src_meta, dst_meta)

    # Verify SHA-256 byte identity
    src_hash = sha256_file(src_payload)
    dst_hash = sha256_file(dst_payload)
    if src_hash != dst_hash:
        raise CorruptedArtifactError(
            f"Backup transfer hash mismatch for {src_payload.name}: src {src_hash} != dst {dst_hash}"
        )

    # Verify metadata hash identity
    src_meta_hash = sha256_file(src_meta)
    dst_meta_hash = sha256_file(dst_meta)
    if src_meta_hash != dst_meta_hash:
        raise CorruptedArtifactError(
            f"Backup metadata transfer hash mismatch for {src_meta.name}: src {src_meta_hash} != dst {dst_meta_hash}"
        )

    return dst_payload, dst_meta, dst_hash


def restore_artifact_from_backup(
    backup_artifact_path: Union[str, Path],
    restore_dir: Union[str, Path],
    expected_parent_hashes: Optional[Dict[str, str]] = None,
    expected_config_hash: Optional[str] = None,
) -> Tuple[bytes, ArtifactMetadata]:
    """Restore an artifact from backup, validating payload digest and provenance lineage."""
    src_payload = Path(backup_artifact_path).resolve()
    dst_dir = Path(restore_dir).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_payload = dst_dir / src_payload.name
    dst_meta = dst_dir / f"{src_payload.name}.meta.json"
    src_meta = src_payload.parent / f"{src_payload.name}.meta.json"

    shutil.copy2(src_payload, dst_payload)
    shutil.copy2(src_meta, dst_meta)

    # Read and verify artifact integrity
    payload_bytes, metadata = read_atomic_artifact(dst_payload)

    # Validate provenance and parent lineage
    validate_provenance(
        metadata,
        expected_parent_hashes=expected_parent_hashes,
        expected_config_hash=expected_config_hash,
    )

    return payload_bytes, metadata
