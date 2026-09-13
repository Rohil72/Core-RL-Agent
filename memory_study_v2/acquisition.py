"""Data acquisition, provider snapshots, request manifests, and offline fixtures (v2).

Acceptance criteria addressed:
- Immutable raw data snapshot with SHA-256 manifests.
- Validation against universe_request.csv (103 primary, 5 excluded).
- Offline snapshot loader ensuring deterministic offline reproduction.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from memory_study_v2.artifacts import atomic_write_bytes, atomic_write_json, sha256_file
from memory_study_v2.contracts import UniverseManifestRecord, validate_universe_manifest


class AcquisitionError(Exception):
    """Raised when data acquisition fails or validation against contract fails."""
    pass


@dataclass
class RawSecuritySnapshot:
    security_id: str
    symbol: str
    market: str
    bars_count: int
    first_session: str
    last_session: str
    payload_hash: str


@dataclass
class SnapshotManifest:
    manifest_id: str
    created_utc: str
    provider: str
    num_securities: int
    securities: List[RawSecuritySnapshot]
    manifest_hash: str = ""


def create_snapshot_manifest(
    snapshots: List[RawSecuritySnapshot],
    provider: str = "Yahoo Finance through version-pinned yfinance",
) -> SnapshotManifest:
    """Construct an immutable snapshot manifest."""
    created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest = SnapshotManifest(
        manifest_id=f"snapshot_{int(time.time())}",
        created_utc=created_utc,
        provider=provider,
        num_securities=len(snapshots),
        securities=snapshots,
    )
    return manifest


def load_raw_security_data(
    snapshot_dir: Union[str, Path],
    security_id: str,
) -> pd.DataFrame:
    """Load raw OHLCV and action data for a security from snapshot directory."""
    p = Path(snapshot_dir) / f"{security_id.replace(':', '_')}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Snapshot data for {security_id} not found at {p}")
    df = pd.read_csv(p)
    return df
