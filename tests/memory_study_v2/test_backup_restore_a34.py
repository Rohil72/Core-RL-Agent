"""Acceptance Test A34: Backup, off-VM retrieval, hash verification, and corruption detection.

Acceptance criteria:
- Completed fold and resumable checkpoint retrieved off-VM with matching hashes.
- Parent hash corruption or payload tampering refuses resume.
"""

from pathlib import Path
import pytest

from memory_study_v2.artifacts import (
    ArtifactMetadata,
    CorruptedArtifactError,
    ProvenanceError,
    atomic_write_bytes,
)
from memory_study_v2.backup import (
    export_artifact_to_backup,
    restore_artifact_from_backup,
)


def test_backup_export_and_restore_hash_match(tmp_path):
    # 1. Create simulated completed fold checkpoint
    staging_dir = tmp_path / "staging"
    backup_dir = tmp_path / "backup_off_vm"
    restore_dir = tmp_path / "restore_dest"

    payload_data = b"SIMULATED_COMPLETED_FOLD_CHECKPOINT_WEIGHTS_AND_STATE_EPOCH_50"
    meta = ArtifactMetadata(
        artifact_id="fold_2020_mlp_seed7_checkpoint.bin",
        parent_hashes={"data_parent": "a" * 64},
        scientific_config_hash="b" * 64,
        completion_state="COMPLETE",
    )
    payload_path, meta_path = atomic_write_bytes(
        staging_dir / "checkpoint.bin",
        payload_data,
        metadata=meta,
    )

    # 2. Export to backup
    dst_payload, dst_meta, exported_hash = export_artifact_to_backup(payload_path, backup_dir)
    assert dst_payload.exists()
    assert dst_meta.exists()

    # 3. Restore to new destination and verify
    restored_bytes, restored_meta = restore_artifact_from_backup(
        dst_payload,
        restore_dir,
        expected_parent_hashes={"data_parent": "a" * 64},
        expected_config_hash="b" * 64,
    )
    assert restored_bytes == payload_data
    assert restored_meta.payload_hash == exported_hash


def test_backup_restore_detects_payload_corruption(tmp_path):
    staging_dir = tmp_path / "staging"
    backup_dir = tmp_path / "backup_off_vm"
    restore_dir = tmp_path / "restore_dest"

    payload_data = b"VALID_STATE_PAYLOAD"
    payload_path, _ = atomic_write_bytes(staging_dir / "model.bin", payload_data)
    dst_payload, _, _ = export_artifact_to_backup(payload_path, backup_dir)

    # Tamper with the backup payload file
    with open(dst_payload, "wb") as f:
        f.write(b"CORRUPTED_PAYLOAD_TAMPERED")

    with pytest.raises(CorruptedArtifactError, match="payload corrupted"):
        restore_artifact_from_backup(dst_payload, restore_dir)


def test_backup_restore_refuses_corrupted_parent_hash(tmp_path):
    staging_dir = tmp_path / "staging"
    backup_dir = tmp_path / "backup_off_vm"
    restore_dir = tmp_path / "restore_dest"

    meta = ArtifactMetadata(
        artifact_id="gated_model.bin",
        parent_hashes={"training_parent": "11" * 32},
        completion_state="COMPLETE",
    )
    payload_path, _ = atomic_write_bytes(
        staging_dir / "gated_model.bin",
        b"VALID_BYTES",
        metadata=meta,
    )
    dst_payload, _, _ = export_artifact_to_backup(payload_path, backup_dir)

    # Expecting different parent hash refuses resume
    with pytest.raises(ProvenanceError, match="Parent hash mismatch for 'training_parent'"):
        restore_artifact_from_backup(
            dst_payload,
            restore_dir,
            expected_parent_hashes={"training_parent": "22" * 32},
        )
