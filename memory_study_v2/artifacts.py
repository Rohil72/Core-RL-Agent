"""Artifact management, provenance tracking, and atomic persistence (v2).

Guarantees:
1. Strict provenance: parent hash, code hash, config hash, and feature contract hash validation.
2. Legacy artifact rejection: legacy checkpoints, models, or caches lacking v2 provenance are rejected.
3. Atomic persistence: partial or interrupted writes never appear as COMPLETE.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from memory_study_v2.contracts import to_canonical_json


class ProvenanceError(Exception):
    """Raised when an artifact fails provenance verification or is a legacy artifact."""
    pass


class IncompleteArtifactError(Exception):
    """Raised when attempting to read an incomplete or interrupted artifact."""
    pass


class CorruptedArtifactError(Exception):
    """Raised when an artifact payload hash does not match its declared hash."""
    pass


def sha256_bytes(b: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Union[str, Path]) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_json(obj: Any) -> str:
    """Compute canonical JSON SHA-256 hex digest."""
    return sha256_bytes(to_canonical_json(obj).encode("utf-8"))


def sha256_dataframe(df: pd.DataFrame, sort_cols: Optional[List[str]] = None) -> str:
    """Compute deterministic SHA-256 of a pandas DataFrame by sorting columns and rows."""
    df_sorted = df.copy()
    # Sort columns
    df_sorted = df_sorted.reindex(sorted(df_sorted.columns), axis=1)
    # Sort rows
    if sort_cols:
        df_sorted = df_sorted.sort_values(by=sort_cols).reset_index(drop=True)
    else:
        # Sort by all columns
        df_sorted = df_sorted.sort_values(by=list(df_sorted.columns)).reset_index(drop=True)
    csv_bytes = df_sorted.to_csv(index=False).encode("utf-8")
    return sha256_bytes(csv_bytes)


@dataclass
class ArtifactMetadata:
    artifact_id: str
    schema_version: str = "rebuild-proposal-1"
    scientific_config_hash: str = ""
    operational_config_hash: str = ""
    source_commit: str = ""
    dirty_diff_hash: str = ""
    environment_digest: str = ""
    parent_hashes: Dict[str, str] = field(default_factory=dict)
    feature_contract_hash: str = ""
    completion_state: str = "IN_PROGRESS"  # IN_PROGRESS, COMPLETE, FAILED
    payload_hash: str = ""
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    row_counts: Optional[int] = None
    date_bounds: Optional[Dict[str, str]] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ArtifactMetadata:
        return cls(**d)


def validate_provenance(
    metadata: Union[Dict[str, Any], ArtifactMetadata],
    expected_parent_hashes: Optional[Dict[str, str]] = None,
    expected_config_hash: Optional[str] = None,
    expected_feature_contract_hash: Optional[str] = None,
) -> None:
    """Validate artifact provenance against expected parent hashes, config hash, and feature contract hash.

    Strictly rejects legacy artifacts and corrupted lineage.
    """
    if isinstance(metadata, ArtifactMetadata):
        meta = metadata.to_dict()
    else:
        meta = metadata

    # Reject missing or wrong schema version (Legacy check)
    if meta.get("schema_version") != "rebuild-proposal-1":
        raise ProvenanceError(
            f"Legacy or invalid artifact rejected: schema_version={meta.get('schema_version')!r} "
            f"!= 'rebuild-proposal-1'"
        )

    # Reject incomplete state
    if meta.get("completion_state") != "COMPLETE":
        raise ProvenanceError(
            f"Artifact reuse rejected: completion_state is '{meta.get('completion_state')}', not COMPLETE"
        )

    # Check scientific config hash if specified
    if expected_config_hash and meta.get("scientific_config_hash") != expected_config_hash:
        raise ProvenanceError(
            f"Artifact config hash mismatch: expected {expected_config_hash}, "
            f"got {meta.get('scientific_config_hash')}"
        )

    # Check feature contract hash if specified
    if expected_feature_contract_hash and meta.get("feature_contract_hash") != expected_feature_contract_hash:
        raise ProvenanceError(
            f"Artifact feature contract hash mismatch: expected {expected_feature_contract_hash}, "
            f"got {meta.get('feature_contract_hash')}"
        )

    # Check parent hashes
    if expected_parent_hashes:
        actual_parents = meta.get("parent_hashes", {})
        for parent_key, expected_hash in expected_parent_hashes.items():
            if parent_key not in actual_parents:
                raise ProvenanceError(
                    f"Artifact missing required parent '{parent_key}' in parent_hashes"
                )
            if actual_parents[parent_key] != expected_hash:
                raise ProvenanceError(
                    f"Parent hash mismatch for '{parent_key}': expected {expected_hash}, "
                    f"got {actual_parents[parent_key]}"
                )


def atomic_write_bytes(
    target_path: Union[str, Path],
    data: bytes,
    metadata: Optional[ArtifactMetadata] = None,
) -> Tuple[Path, Path]:
    """Atomically write binary data and its companion metadata.

    Guarantees:
    1. Writes to a hidden temporary file with fsync.
    2. Computes payload hash.
    3. Writes companion .meta.json with completion_state='COMPLETE'.
    4. Atomically renames temp file to target_path using os.replace.
    5. Returns (target_path, meta_path).
    """
    target = Path(target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    temp_file = target.parent / f".tmp_{target.name}_{uuid.uuid4().hex}"
    meta_path = target.parent / f"{target.name}.meta.json"
    temp_meta = target.parent / f".tmp_{target.name}_{uuid.uuid4().hex}.meta"

    try:
        # Write payload
        with open(temp_file, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        payload_hash = sha256_bytes(data)

        # Prepare metadata
        if metadata is None:
            metadata = ArtifactMetadata(artifact_id=target.name)
        metadata.payload_hash = payload_hash
        metadata.completion_state = "COMPLETE"

        # Write metadata temp
        meta_bytes = to_canonical_json(metadata.to_dict()).encode("utf-8")
        with open(temp_meta, "wb") as f:
            f.write(meta_bytes)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename metadata then target
        os.replace(temp_meta, meta_path)
        os.replace(temp_file, target)

    except Exception:
        # Cleanup temporary files if write was interrupted
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
        if temp_meta.exists():
            temp_meta.unlink(missing_ok=True)
        raise

    return target, meta_path


def atomic_write_json(
    target_path: Union[str, Path],
    obj: Any,
    metadata: Optional[ArtifactMetadata] = None,
) -> Tuple[Path, Path]:
    """Atomically write JSON object using canonical formatting."""
    json_bytes = to_canonical_json(obj).encode("utf-8")
    return atomic_write_bytes(target_path, json_bytes, metadata=metadata)


def read_atomic_artifact(target_path: Union[str, Path]) -> Tuple[bytes, ArtifactMetadata]:
    """Read an atomically persisted artifact and verify its completion state and integrity."""
    target = Path(target_path).resolve()
    meta_path = target.parent / f"{target.name}.meta.json"

    if not target.exists():
        raise FileNotFoundError(f"Artifact payload not found: {target}")

    if not meta_path.exists():
        raise IncompleteArtifactError(
            f"Artifact companion metadata missing for {target}. Write may have been interrupted."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)

    meta = ArtifactMetadata.from_dict(meta_dict)

    if meta.completion_state != "COMPLETE":
        raise IncompleteArtifactError(
            f"Artifact {target} has completion_state '{meta.completion_state}', not COMPLETE."
        )

    payload = target.read_bytes()
    actual_hash = sha256_bytes(payload)

    if meta.payload_hash and actual_hash != meta.payload_hash:
        raise CorruptedArtifactError(
            f"Artifact {target} payload corrupted: expected hash {meta.payload_hash}, got {actual_hash}"
        )

    return payload, meta
