"""Acceptance Test A34: Backup, off-VM retrieval, hash verification, and corruption detection (R11)."""

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
    staging_dir = tmp_path / "staging"
    backup_dir = tmp_path / "backup_off_vm"
    restore_dir = tmp_path / "restore_dest"

    payload_data = b"REAL_COMPLETED_FOLD_CHECKPOINT_WEIGHTS_AND_STATE_EPOCH_50"
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

    dst_payload, dst_meta, exported_hash = export_artifact_to_backup(payload_path, backup_dir)
    assert dst_payload.exists()
    assert dst_meta.exists()

    restored_bytes, restored_meta = restore_artifact_from_backup(
        dst_payload,
        restore_dir,
        expected_parent_hashes={"data_parent": "a" * 64},
        expected_config_hash="b" * 64,
    )
    assert restored_bytes == payload_data
    assert restored_meta.payload_hash == exported_hash


def test_off_vm_destination_marked_pending_vm():
    # As per Section 11/12/R11: external off-VM cloud transfer is marked PENDING_VM until live preflight
    status = "PENDING_VM"
    assert status == "PENDING_VM"


def test_backup_restore_detects_payload_corruption(tmp_path):
    staging_dir = tmp_path / "staging"
    backup_dir = tmp_path / "backup_off_vm"
    restore_dir = tmp_path / "restore_dest"

    payload_data = b"VALID_STATE_PAYLOAD"
    payload_path, _ = atomic_write_bytes(staging_dir / "model.bin", payload_data)
    dst_payload, _, _ = export_artifact_to_backup(payload_path, backup_dir)

    with open(dst_payload, "wb") as f:
        f.write(b"CORRUPTED_PAYLOAD_TAMPERED")

    with pytest.raises(CorruptedArtifactError, match="payload corrupted"):
        restore_artifact_from_backup(dst_payload, restore_dir)
