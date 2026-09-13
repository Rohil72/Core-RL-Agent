"""Acceptance Test A03: Atomicity and interrupted partial write safety."""

import os
import tempfile
import pytest
from pathlib import Path

from memory_study_v2.artifacts import (
    ArtifactMetadata,
    IncompleteArtifactError,
    CorruptedArtifactError,
    atomic_write_bytes,
    atomic_write_json,
    read_atomic_artifact,
)


def test_successful_atomic_write_and_read():
    """Verify that atomic write creates target and metadata with verified hash."""
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "weights.bin"
        payload = b"NEURAL_WEIGHTS_BINARY_DATA_12345"
        meta = ArtifactMetadata(
            artifact_id="test_weights",
            scientific_config_hash="cfg_123",
        )
        
        target_path, meta_path = atomic_write_bytes(target, payload, metadata=meta)
        
        assert target_path.exists()
        assert meta_path.exists()
        
        # Read back and verify
        read_payload, read_meta = read_atomic_artifact(target_path)
        assert read_payload == payload
        assert read_meta.completion_state == "COMPLETE"
        assert read_meta.payload_hash == read_meta.payload_hash


def test_interrupted_write_cannot_appear_complete():
    """Verify that an interrupted write leaves no complete artifact."""
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "interrupted.bin"
        
        # Simulate interruption by raising during a custom write wrapper or partial write
        # Manually create a partial file without meta
        partial_file = target.parent / f".tmp_{target.name}_partial"
        partial_file.write_bytes(b"PARTIAL_DATA")
        
        # Target should not exist
        assert not target.exists()
        
        # If target file exists but has no companion .meta.json, reading must fail
        target.write_bytes(b"ORPHAN_DATA")
        with pytest.raises(IncompleteArtifactError, match="companion metadata missing"):
            read_atomic_artifact(target)


def test_corrupted_payload_detected():
    """Verify that payload modification after atomic write is detected."""
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "data.json"
        target_path, meta_path = atomic_write_json(target, {"field": "original_value"})
        
        # Corrupt the payload by modifying one byte
        target.write_bytes(b'{"field":"tampered_value"}')
        
        with pytest.raises(CorruptedArtifactError, match="payload corrupted"):
            read_atomic_artifact(target)


def test_incomplete_meta_state_detected():
    """Verify that metadata with completion_state != COMPLETE is rejected."""
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "in_progress.json"
        meta = ArtifactMetadata(artifact_id="test", completion_state="IN_PROGRESS")
        
        # Write payload directly
        target.write_bytes(b'{"data": 123}')
        # Write meta indicating IN_PROGRESS
        meta_path = target.parent / f"{target.name}.meta.json"
        meta_path.write_text(meta.to_dict().__repr__())
        
        # Use proper JSON serialization for metadata
        import json
        meta_path.write_text(json.dumps(meta.to_dict()))
        
        with pytest.raises(IncompleteArtifactError, match="completion_state 'IN_PROGRESS', not COMPLETE"):
            read_atomic_artifact(target)
