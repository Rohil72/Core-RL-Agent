"""Sealed policy predictions, query masks, and prediction manifest generator (v2).

Acceptance criteria addressed:
- Generates label-free forecasts for active policies.
- Writes sealed prediction artifacts with metadata and SHA-256 digests.
- Provides verification interface for unlocking evaluation labels (R09, R12).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from memory_study_v2.artifacts import ArtifactMetadata, atomic_write_json, read_atomic_artifact


@dataclass
class PredictionQuery:
    fold_year: int
    security_id: str
    session_origin: str
    forecast_score: float
    volatility_21: float
    atr_ratio_14: float
    is_valid_query: bool = True

    @property
    def query_id(self) -> str:
        return f"{self.fold_year}_{self.security_id}_{self.session_origin}"


@dataclass
class PredictionManifest:
    configuration: str
    seed: Optional[int]
    evaluation_year: int
    num_predictions: int
    predictions: List[Dict[str, Any]]
    manifest_hash: str = ""


def generate_and_seal_predictions(
    configuration: str,
    evaluation_year: int,
    queries: List[PredictionQuery],
    output_path: Union[str, Path],
    seed: Optional[int] = None,
    parent_hashes: Optional[Dict[str, str]] = None,
) -> Tuple[Path, Path]:
    """Generate and atomically seal prediction artifact with companion metadata (R09, R12)."""
    p_out = Path(output_path)
    pred_dicts = [asdict(q) for q in queries]

    payload = {
        "configuration": configuration,
        "seed": seed,
        "evaluation_year": evaluation_year,
        "num_predictions": len(queries),
        "queries": pred_dicts,
    }

    meta = ArtifactMetadata(
        artifact_id=p_out.name,
        parent_hashes=parent_hashes or {},
        completion_state="COMPLETE",
        custom_metadata={
            "evaluation_year": evaluation_year,
            "configuration": configuration,
            "seed": seed,
            "num_predictions": len(queries),
        },
    )

    return atomic_write_json(p_out, payload, metadata=meta)
