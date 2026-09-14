"""Export and Package Production Checkpoints with Cryptographic Digests (R06, C4).

Bundles trained backbone checkpoints (MLP, Transformer) across walk-forward folds
and seeds, computes SHA-256 digests, extracts training metadata (epoch, best_loss,
macro_step, config_hash), and generates a verified checkpoint_manifest.json.

Usage:
  python scripts/export_production_checkpoints.py --checkpoint-dir rebuild_plan/connected_pilot_out --output-dir rebuild_plan/exported_checkpoints
  python scripts/export_production_checkpoints.py --verify-manifest rebuild_plan/exported_checkpoints/checkpoint_manifest.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional
import zipfile

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
except ImportError:
    torch = None

from memory_study_v2.contracts import to_canonical_json


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 digest of a file in 64KB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def inspect_checkpoint(path: Path) -> Dict[str, Any]:
    """Inspect checkpoint structure and extract metadata without loading model to GPU."""
    if torch is None:
        raise RuntimeError("PyTorch is required to inspect checkpoints.")

    # Load on CPU
    data = torch.load(path, map_location="cpu", weights_only=False)

    meta: Dict[str, Any] = {
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        "sha256": compute_file_sha256(path),
    }

    if hasattr(data, "epoch"):
        # TrainingState dataclass
        meta["epoch"] = data.epoch
        meta["macro_step"] = data.macro_step
        meta["micro_step"] = data.micro_step
        raw_loss = data.best_loss
        meta["best_loss"] = float(raw_loss) if (raw_loss is not None and math.isfinite(raw_loss)) else None
        meta["best_epoch"] = data.best_epoch
        meta["config_hash"] = getattr(data, "config_hash", None)
        meta["has_model_state"] = bool(data.model_state)
        meta["has_optimizer_state"] = bool(data.optimizer_state)
        meta["num_params_tensors"] = len(data.model_state) if data.model_state else 0
    elif isinstance(data, dict):
        meta["epoch"] = data.get("epoch")
        meta["macro_step"] = data.get("macro_step")
        meta["micro_step"] = data.get("micro_step")
        raw_loss = data.get("best_loss")
        meta["best_loss"] = float(raw_loss) if (raw_loss is not None and math.isfinite(raw_loss)) else None
        meta["best_epoch"] = data.get("best_epoch")
        meta["config_hash"] = data.get("config_hash")
        meta["has_model_state"] = "model_state" in data or "state_dict" in data
        meta["has_optimizer_state"] = "optimizer_state" in data
    else:
        meta["type"] = type(data).__name__

    return meta


def export_checkpoints(
    source_dirs: List[Path],
    output_dir: Path,
    copy_files: bool = True,
    archive_format: Optional[str] = None,
) -> Dict[str, Any]:
    """Scan, inspect, bundle, and generate manifest for checkpoints."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_found: List[Path] = []

    for s_dir in source_dirs:
        if not s_dir.exists():
            continue
        # Scan for .pt and .pth files recursively
        for ext in ("*.pt", "*.pth"):
            checkpoints_found.extend(s_dir.rglob(ext))

    checkpoints_found = sorted(list(set(checkpoints_found)))
    if not checkpoints_found:
        print(f"Warning: No checkpoint files found in {[str(d) for d in source_dirs]}")

    entries: List[Dict[str, Any]] = []
    exported_dir = output_dir / "checkpoints"
    if copy_files and checkpoints_found:
        exported_dir.mkdir(parents=True, exist_ok=True)

    for chk_path in checkpoints_found:
        info = inspect_checkpoint(chk_path)
        # Compute relative or canonical target path
        if copy_files:
            rel_name = chk_path.name
            target_path = exported_dir / f"{chk_path.parent.name}_{rel_name}"
            shutil.copy2(chk_path, target_path)
            info["exported_path"] = str(target_path.relative_to(output_dir))
            info["original_path"] = str(chk_path)
        else:
            info["source_path"] = str(chk_path)
        entries.append(info)

    manifest: Dict[str, Any] = {
        "manifest_type": "PRODUCTION_CHECKPOINT_EXPORT_MANIFEST",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_checkpoints": len(entries),
        "total_bytes": sum(e["file_size_bytes"] for e in entries),
        "checkpoints": entries,
    }

    manifest_path = output_dir / "checkpoint_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(to_canonical_json(manifest))

    manifest_sha = compute_file_sha256(manifest_path)
    print(f"Exported {len(entries)} checkpoints to {output_dir}")
    print(f"Manifest written: {manifest_path} (SHA-256: {manifest_sha})")

    # Optional archive creation
    archive_path = None
    if archive_format in ("tar.gz", "tar"):
        archive_path = output_dir / "production_checkpoints.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(manifest_path, arcname="checkpoint_manifest.json")
            if copy_files and exported_dir.exists():
                tar.add(exported_dir, arcname="checkpoints")
        print(f"Archive created: {archive_path} (SHA-256: {compute_file_sha256(archive_path)})")
    elif archive_format == "zip":
        archive_path = output_dir / "production_checkpoints.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_path, arcname="checkpoint_manifest.json")
            if copy_files and exported_dir.exists():
                for f in exported_dir.rglob("*"):
                    zf.write(f, arcname=str(f.relative_to(output_dir)))
        print(f"Archive created: {archive_path} (SHA-256: {compute_file_sha256(archive_path)})")

    return {
        "status": "EXPORT_SUCCESS",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "checkpoint_count": len(entries),
        "archive_path": str(archive_path) if archive_path else None,
    }


def verify_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Verify all checkpoints listed in manifest match their recorded SHA-256 digests."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    parent_dir = manifest_path.parent
    checkpoints = manifest.get("checkpoints", [])
    results = []
    all_ok = True

    for c in checkpoints:
        rel_path = c.get("exported_path") or c.get("source_path") or c.get("file_name")
        full_path = parent_dir / rel_path
        if not full_path.exists():
            results.append({"path": str(full_path), "status": "MISSING"})
            all_ok = False
            continue

        actual_sha = compute_file_sha256(full_path)
        expected_sha = c.get("sha256")
        if actual_sha == expected_sha:
            results.append({"path": str(full_path), "status": "VERIFIED_MATCH", "sha256": actual_sha})
        else:
            results.append({
                "path": str(full_path),
                "status": "DIGEST_MISMATCH",
                "actual": actual_sha,
                "expected": expected_sha,
            })
            all_ok = False

    res = {
        "status": "ALL_VERIFIED" if all_ok else "VERIFICATION_FAILED",
        "verified_count": len([r for r in results if r["status"] == "VERIFIED_MATCH"]),
        "total_count": len(checkpoints),
        "details": results,
    }
    print(f"Manifest Verification: {res['status']} ({res['verified_count']}/{res['total_count']} verified)")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and Package Production Checkpoints")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        nargs="+",
        default=[
            REPO_ROOT / "rebuild_plan" / "connected_pilot_out",
            REPO_ROOT / "checkpoints",
        ],
        help="One or more directories containing model checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "rebuild_plan" / "exported_checkpoints",
        help="Directory to write exported manifest and checkpoint bundle",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        default=False,
        help="Do not copy checkpoint files, only generate manifest from source locations",
    )
    parser.add_argument(
        "--archive",
        choices=["tar.gz", "zip"],
        default=None,
        help="Optionally package into a compressed archive (.tar.gz or .zip)",
    )
    parser.add_argument(
        "--verify-manifest",
        type=Path,
        default=None,
        help="Path to an existing checkpoint_manifest.json to verify cryptographic integrity",
    )
    args = parser.parse_args()

    if args.verify_manifest:
        rep = verify_manifest(args.verify_manifest)
        if rep["status"] != "ALL_VERIFIED":
            sys.exit(1)
        sys.exit(0)

    export_checkpoints(
        source_dirs=args.checkpoint_dir,
        output_dir=args.output_dir,
        copy_files=not args.no_copy,
        archive_format=args.archive,
    )


if __name__ == "__main__":
    main()
