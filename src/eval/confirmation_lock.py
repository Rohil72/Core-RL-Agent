from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def sha256_file(path: str | Path) -> str:
    """Hash an immutable research input or promoted model artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_confirmation_lock(
    path: str | Path,
    artifacts: Iterable[str | Path],
    markets: Iterable[str],
    confirmation_year: int = 2025,
) -> dict[str, object]:
    """Seal final inputs and refuse to overwrite an already executed confirmation."""
    destination = Path(path)
    if destination.exists():
        current = json.loads(destination.read_text(encoding="utf-8"))
        if current.get("executed_at"):
            raise FileExistsError("The final confirmation has already been executed and cannot be replaced.")
    files = []
    for artifact in artifacts:
        item = Path(artifact)
        if not item.exists():
            raise FileNotFoundError(item)
        files.append({"path": str(item.resolve()), "sha256": sha256_file(item), "bytes": item.stat().st_size})
    lock = {
        "confirmation_year": confirmation_year,
        "markets": list(markets),
        "artifacts": files,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "executed_at": None,
        "status": "locked_not_executed",
    }
    if destination.exists():
        comparable_current = {
            key: current.get(key)
            for key in ("confirmation_year", "markets", "artifacts", "status")
        }
        comparable_requested = {
            key: lock.get(key)
            for key in ("confirmation_year", "markets", "artifacts", "status")
        }
        if comparable_current != comparable_requested:
            raise FileExistsError(
                "An unexecuted confirmation lock already exists with different immutable inputs."
            )
        return current
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


def mark_confirmation_executed(
    path: str | Path,
    result_files: Iterable[str | Path],
    *,
    idempotent: bool = False,
) -> dict[str, object]:
    """Close the one-shot lock after result artifacts have been written."""
    destination = Path(path)
    lock = json.loads(destination.read_text(encoding="utf-8"))
    requested_results = [
        {"path": str(Path(item).resolve()), "sha256": sha256_file(item)} for item in result_files
    ]
    if lock.get("executed_at"):
        if idempotent and lock.get("results") == requested_results:
            return lock
        raise FileExistsError("The final confirmation has already been executed with different results.")
    lock["results"] = requested_results
    lock["executed_at"] = datetime.now(timezone.utc).isoformat()
    lock["status"] = "executed_frozen"
    destination.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock
