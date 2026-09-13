"""Acceptance Test A02: Provenance and lineage validation."""

import pytest
from memory_study_v2.artifacts import (
    ArtifactMetadata,
    ProvenanceError,
    validate_provenance,
    sha256_bytes,
)


def test_valid_provenance_passes():
    """Verify valid provenance matching all expected hashes passes."""
    meta = ArtifactMetadata(
        artifact_id="features_fold_2020",
        schema_version="rebuild-proposal-1",
        scientific_config_hash="cfg_hash_123",
        feature_contract_hash="feat_hash_456",
        parent_hashes={"canonical_bars": "bars_hash_789"},
        completion_state="COMPLETE",
    )
    # Should pass without error
    validate_provenance(
        meta,
        expected_parent_hashes={"canonical_bars": "bars_hash_789"},
        expected_config_hash="cfg_hash_123",
        expected_feature_contract_hash="feat_hash_456",
    )


def test_parent_hash_mismatch_rejects():
    """Verify that a changed parent hash rejects checkpoint/cache reuse."""
    meta = ArtifactMetadata(
        artifact_id="checkpoint_mlp_fold2020_seed7",
        schema_version="rebuild-proposal-1",
        scientific_config_hash="cfg_hash_123",
        feature_contract_hash="feat_hash_456",
        parent_hashes={"training_data": "original_data_hash"},
        completion_state="COMPLETE",
    )
    with pytest.raises(ProvenanceError, match="Parent hash mismatch for 'training_data'"):
        validate_provenance(
            meta,
            expected_parent_hashes={"training_data": "tampered_data_hash"},
        )


def test_feature_contract_hash_mismatch_rejects():
    """Verify that a changed feature contract rejects reuse."""
    meta = ArtifactMetadata(
        artifact_id="bank_fold_2020",
        schema_version="rebuild-proposal-1",
        feature_contract_hash="old_feature_contract_hash",
        completion_state="COMPLETE",
    )
    with pytest.raises(ProvenanceError, match="Artifact feature contract hash mismatch"):
        validate_provenance(
            meta,
            expected_feature_contract_hash="new_feature_contract_hash",
        )


def test_missing_parent_hash_rejects():
    """Verify that a missing parent hash dependency raises ProvenanceError."""
    meta = ArtifactMetadata(
        artifact_id="model_ckpt",
        schema_version="rebuild-proposal-1",
        parent_hashes={"features": "f_hash"},
        completion_state="COMPLETE",
    )
    with pytest.raises(ProvenanceError, match="missing required parent 'raw_data'"):
        validate_provenance(
            meta,
            expected_parent_hashes={"features": "f_hash", "raw_data": "r_hash"},
        )


def test_legacy_artifact_rejected():
    """Verify that legacy checkpoints or caches lacking v2 provenance are strictly rejected."""
    # Legacy checkpoint missing schema_version or using legacy version
    legacy_meta_dict = {
        "artifact_id": "legacy_transformer_model",
        "model_type": "transformer",
        "step": 5000,
        "completion_state": "COMPLETE",
    }
    with pytest.raises(ProvenanceError, match="Legacy or invalid artifact rejected"):
        validate_provenance(legacy_meta_dict)


def test_incomplete_state_rejects_reuse():
    """Verify that artifacts marked IN_PROGRESS or FAILED cannot be reused."""
    meta_in_prog = ArtifactMetadata(
        artifact_id="fold_2022_features",
        schema_version="rebuild-proposal-1",
        completion_state="IN_PROGRESS",
    )
    with pytest.raises(ProvenanceError, match="completion_state is 'IN_PROGRESS', not COMPLETE"):
        validate_provenance(meta_in_prog)

    meta_failed = ArtifactMetadata(
        artifact_id="fold_2022_features",
        schema_version="rebuild-proposal-1",
        completion_state="FAILED",
    )
    with pytest.raises(ProvenanceError, match="completion_state is 'FAILED', not COMPLETE"):
        validate_provenance(meta_failed)
