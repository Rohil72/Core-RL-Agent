"""Shared persistent security feature cache (Phase 3).

Computes full-history features, total-return bars, target labels, and validity flags
once per security across all available history, caching to disk in NPZ format.
Walk-forward folds then slice rows directly with zero redundant calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from memory_study_v2.canonical_data import (
    align_to_venue_calendar,
    build_total_return_bars,
    validate_raw_bars,
)
from memory_study_v2.contracts import EXPECTED_FEATURES_ORDERED
from memory_study_v2.features import compute_technical_features
from memory_study_v2.labels import compute_target_labels
from memory_study_v2.venue_calendar import VenueCalendar, get_market_venue_calendar


FEATURE_SPEC_ID = "TECHNICAL_23_FEATURES_V2"
TARGET_SPEC_ID = "CLOSE63_MODEL_V1"


@dataclass
class SecurityFeatureCache:
    """In-memory or serialized persistent cache of full-history security data."""
    security_id: str
    sessions: np.ndarray          # 1D '<U10'
    raw_open: np.ndarray          # 1D float64
    raw_high: np.ndarray          # 1D float64
    raw_low: np.ndarray           # 1D float64
    raw_close: np.ndarray         # 1D float64
    model_open: np.ndarray        # 1D float64
    model_high: np.ndarray        # 1D float64
    model_low: np.ndarray         # 1D float64
    model_close: np.ndarray       # 1D float64
    volume: np.ndarray            # 1D float64
    raw_features: np.ndarray      # 2D (T, 23) float64
    target_63_legacy: np.ndarray  # 1D float64
    target_executable: np.ndarray # 1D float64
    session_ordinals: np.ndarray  # 1D int64
    validity_flags: np.ndarray    # 1D bool
    bar_status: np.ndarray        # 1D str
    file_sha256: str
    calendar_sha256: str
    feature_valid_mask: Optional[np.ndarray] = None # 1D bool
    feature_spec_id: str = FEATURE_SPEC_ID
    target_spec_id: str = TARGET_SPEC_ID
    segment_ids: Optional[np.ndarray] = None        # 1D int64

    def save(self, cache_dest: Union[str, Path]) -> None:
        """Save cache record atomically as .npy memmaps plus manifest.json, or uncompressed .npz."""
        dest = Path(cache_dest)
        if dest.suffix == ".npz":
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = dest.with_suffix(".tmp.npz")
            np.savez(
                tmp_file,
                security_id=np.array(self.security_id),
                sessions=self.sessions,
                raw_open=self.raw_open,
                raw_high=self.raw_high,
                raw_low=self.raw_low,
                raw_close=self.raw_close,
                model_open=self.model_open,
                model_high=self.model_high,
                model_low=self.model_low,
                model_close=self.model_close,
                volume=self.volume,
                raw_features=self.raw_features,
                target_63_legacy=self.target_63_legacy,
                target_executable=self.target_executable,
                session_ordinals=self.session_ordinals,
                validity_flags=self.validity_flags,
                bar_status=self.bar_status,
                feature_valid_mask=self.feature_valid_mask if self.feature_valid_mask is not None else self.validity_flags,
                segment_ids=self.segment_ids if self.segment_ids is not None else np.zeros(len(self.sessions), dtype=np.int64),
                file_sha256=np.array(self.file_sha256),
                calendar_sha256=np.array(self.calendar_sha256),
                feature_spec_id=np.array(self.feature_spec_id),
                target_spec_id=np.array(self.target_spec_id),
            )
            if os.name == "nt" and dest.exists():
                dest.unlink()
            tmp_file.rename(dest)
            return

        # Directory-based .npy memmaps + JSON manifest (Correction 1)
        dest.mkdir(parents=True, exist_ok=True)
        import json
        seg_arr = self.segment_ids if self.segment_ids is not None else np.zeros(len(self.sessions), dtype=np.int64)
        manifest = {
            "security_id": str(self.security_id),
            "file_sha256": str(self.file_sha256),
            "calendar_sha256": str(self.calendar_sha256),
            "feature_spec_id": str(self.feature_spec_id),
            "target_spec_id": str(self.target_spec_id),
            "arrays": {
                "sessions": {"dtype": str(self.sessions.dtype), "shape": list(self.sessions.shape)},
                "raw_open": {"dtype": str(self.raw_open.dtype), "shape": list(self.raw_open.shape)},
                "raw_high": {"dtype": str(self.raw_high.dtype), "shape": list(self.raw_high.shape)},
                "raw_low": {"dtype": str(self.raw_low.dtype), "shape": list(self.raw_low.shape)},
                "raw_close": {"dtype": str(self.raw_close.dtype), "shape": list(self.raw_close.shape)},
                "model_open": {"dtype": str(self.model_open.dtype), "shape": list(self.model_open.shape)},
                "model_high": {"dtype": str(self.model_high.dtype), "shape": list(self.model_high.shape)},
                "model_low": {"dtype": str(self.model_low.dtype), "shape": list(self.model_low.shape)},
                "model_close": {"dtype": str(self.model_close.dtype), "shape": list(self.model_close.shape)},
                "volume": {"dtype": str(self.volume.dtype), "shape": list(self.volume.shape)},
                "raw_features": {"dtype": str(self.raw_features.dtype), "shape": list(self.raw_features.shape)},
                "target_63_legacy": {"dtype": str(self.target_63_legacy.dtype), "shape": list(self.target_63_legacy.shape)},
                "target_executable": {"dtype": str(self.target_executable.dtype), "shape": list(self.target_executable.shape)},
                "session_ordinals": {"dtype": str(self.session_ordinals.dtype), "shape": list(self.session_ordinals.shape)},
                "validity_flags": {"dtype": str(self.validity_flags.dtype), "shape": list(self.validity_flags.shape)},
                "bar_status": {"dtype": str(self.bar_status.dtype), "shape": list(self.bar_status.shape)},
                "feature_valid_mask": {"dtype": str((self.feature_valid_mask if self.feature_valid_mask is not None else self.validity_flags).dtype), "shape": list(self.validity_flags.shape)},
                "segment_ids": {"dtype": str(seg_arr.dtype), "shape": list(seg_arr.shape)},
            }
        }
        for name in manifest["arrays"]:
            arr = getattr(self, name, None)
            if arr is None and name == "feature_valid_mask":
                arr = self.validity_flags
            elif arr is None and name == "segment_ids":
                arr = seg_arr
            np.save(dest / f"{name}.npy", arr)

        manifest_tmp = dest / "manifest.tmp.json"
        with open(manifest_tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        if os.name == "nt" and (dest / "manifest.json").exists():
            (dest / "manifest.json").unlink()
        manifest_tmp.rename(dest / "manifest.json")

    @classmethod
    def load(cls, cache_path: Union[str, Path], mmap_mode: Optional[str] = "r") -> SecurityFeatureCache:
        """Load cache record from .npy memmaps directory or fallback to .npz."""
        p = Path(cache_path)
        manifest_file = p / "manifest.json" if p.is_dir() else (p.parent / f"{p.stem}" / "manifest.json" if (p.parent / f"{p.stem}" / "manifest.json").exists() else None)
        if manifest_file and manifest_file.exists():
            import json
            sec_dir = manifest_file.parent
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            def _load_arr(name: str):
                npy_path = sec_dir / f"{name}.npy"
                return np.load(npy_path, mmap_mode=mmap_mode)

            return cls(
                security_id=str(manifest["security_id"]),
                sessions=_load_arr("sessions").astype('<U10'),
                raw_open=_load_arr("raw_open"),
                raw_high=_load_arr("raw_high"),
                raw_low=_load_arr("raw_low"),
                raw_close=_load_arr("raw_close"),
                model_open=_load_arr("model_open"),
                model_high=_load_arr("model_high"),
                model_low=_load_arr("model_low"),
                model_close=_load_arr("model_close"),
                volume=_load_arr("volume"),
                raw_features=_load_arr("raw_features"),
                target_63_legacy=_load_arr("target_63_legacy"),
                target_executable=_load_arr("target_executable"),
                session_ordinals=_load_arr("session_ordinals"),
                validity_flags=_load_arr("validity_flags"),
                bar_status=_load_arr("bar_status").astype(str),
                file_sha256=str(manifest["file_sha256"]),
                calendar_sha256=str(manifest["calendar_sha256"]),
                feature_valid_mask=_load_arr("feature_valid_mask") if (sec_dir / "feature_valid_mask.npy").exists() else None,
                feature_spec_id=str(manifest.get("feature_spec_id", FEATURE_SPEC_ID)),
                target_spec_id=str(manifest.get("target_spec_id", TARGET_SPEC_ID)),
                segment_ids=_load_arr("segment_ids") if (sec_dir / "segment_ids.npy").exists() else None,
            )

        # NPZ fallback
        with np.load(p, allow_pickle=False) as data:
            return cls(
                security_id=str(data["security_id"]),
                sessions=data["sessions"].astype('<U10'),
                raw_open=data["raw_open"].astype(np.float64),
                raw_high=data["raw_high"].astype(np.float64),
                raw_low=data["raw_low"].astype(np.float64),
                raw_close=data["raw_close"].astype(np.float64),
                model_open=data["model_open"].astype(np.float64),
                model_high=data["model_high"].astype(np.float64),
                model_low=data["model_low"].astype(np.float64),
                model_close=data["model_close"].astype(np.float64),
                volume=data["volume"].astype(np.float64),
                raw_features=data["raw_features"].astype(np.float64),
                target_63_legacy=data["target_63_legacy"].astype(np.float64),
                target_executable=data["target_executable"].astype(np.float64),
                session_ordinals=data["session_ordinals"].astype(np.int64),
                validity_flags=data["validity_flags"].astype(bool),
                bar_status=data["bar_status"].astype(str),
                file_sha256=str(data["file_sha256"]),
                calendar_sha256=str(data["calendar_sha256"]),
                feature_valid_mask=data["feature_valid_mask"].astype(bool) if "feature_valid_mask" in data else data["validity_flags"].astype(bool),
                feature_spec_id=str(data["feature_spec_id"]),
                target_spec_id=str(data["target_spec_id"]),
                segment_ids=data["segment_ids"].astype(np.int64) if "segment_ids" in data else None,
            )

    def to_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convert cache arrays into (val_df, tr_df, feats_df, labels_df) matching legacy contracts."""
        seg_ids = self.segment_ids if self.segment_ids is not None else np.zeros(len(self.sessions), dtype=np.int64)
        val_df = pd.DataFrame({
            "session": self.sessions,
            "open": self.raw_open,
            "high": self.raw_high,
            "low": self.raw_low,
            "close": self.raw_close,
            "volume": self.volume,
            "bar_status": self.bar_status,
            "is_valid_bar": self.validity_flags,
            "segment_id": seg_ids,
        })
        tr_df = pd.DataFrame({
            "session": self.sessions,
            "raw_open": self.raw_open,
            "raw_high": self.raw_high,
            "raw_low": self.raw_low,
            "raw_close": self.raw_close,
            "raw_volume": self.volume,
            "model_open": self.model_open,
            "model_high": self.model_high,
            "model_low": self.model_low,
            "model_close": self.model_close,
            "tr_open": self.model_open,
            "tr_high": self.model_high,
            "tr_low": self.model_low,
            "tr_close": self.model_close,
            "volume": self.volume,
            "normalized_volume": self.volume,
            "is_valid_bar": self.validity_flags,
            "segment_id": seg_ids,
        })
        feats_dict = {"session": self.sessions}
        for idx, feat_name in enumerate(EXPECTED_FEATURES_ORDERED):
            feats_dict[feat_name] = self.raw_features[:, idx]
        feats_dict["valid_mask"] = self.feature_valid_mask if getattr(self, "feature_valid_mask", None) is not None else self.validity_flags
        feats_df = pd.DataFrame(feats_dict)

        labels_df = pd.DataFrame({
            "session_origin": self.sessions,
            "target_value": self.target_63_legacy,
            "valid_63": np.isfinite(self.target_63_legacy),
        })
        return val_df, tr_df, feats_df, labels_df


def compute_security_cache(
    parquet_path: Union[str, Path],
    sec_cal: VenueCalendar,
    end_year: int = 2026,
) -> SecurityFeatureCache:
    """Compute full-history feature and label cache for a single security."""
    parquet_path = Path(parquet_path)
    sec_id = parquet_path.stem
    file_bytes = parquet_path.read_bytes()
    file_sha = hashlib.sha256(file_bytes).hexdigest()
    cal_sha = getattr(sec_cal, "schedule_hash", None) or hashlib.sha256("".join(sec_cal.sessions).encode()).hexdigest()

    venue_sessions = sec_cal.sessions_in_range("2010-01-01", f"{end_year}-12-31")

    df_raw = pd.read_parquet(parquet_path).reset_index()
    rename_dict = {}
    for c in df_raw.columns:
        if c.lower() in ("date", "session", "index", "timestamp"):
            rename_dict[c] = "session"
    df_raw = df_raw.rename(columns=rename_dict)
    if "session" not in df_raw.columns:
        df_raw["session"] = df_raw.iloc[:, 0].astype(str).str.slice(0, 10)
    else:
        df_raw["session"] = df_raw["session"].astype(str).str.slice(0, 10)

    df_aligned = align_to_venue_calendar(df_raw, venue_sessions)
    val_df = validate_raw_bars(df_aligned, sec_id, quote_unit=1.0, reject_material=True)
    tr_df = build_total_return_bars(val_df, actions=[], quote_unit=1.0)
    s_min = val_df["session"].min()
    s_max = val_df["session"].max()
    sched_val_df = sec_cal.reindex_to_schedule(val_df, (s_min, s_max))

    feats_df = compute_technical_features(tr_df)
    labels_df = compute_target_labels(tr_df, venue_sessions=sec_cal.sessions)

    # Align arrays to feats_df sessions (the canonical feature rows)
    sessions = feats_df["session"].to_numpy(dtype='<U10')
    T = len(sessions)

    # Map sessions to ordinals on the venue calendar
    sess_to_ord = {s: i for i, s in enumerate(sec_cal.sessions)}
    session_ordinals = np.array([sess_to_ord.get(s, 0) for s in sessions], dtype=np.int64)

    # Extract raw/model OHLCV and volume
    raw_open = tr_df["raw_open"].to_numpy(dtype=np.float64)
    raw_high = tr_df["raw_high"].to_numpy(dtype=np.float64)
    raw_low = tr_df["raw_low"].to_numpy(dtype=np.float64)
    raw_close = tr_df["raw_close"].to_numpy(dtype=np.float64)

    model_open = (tr_df["tr_open"] if "tr_open" in tr_df.columns else (tr_df["model_open"] if "model_open" in tr_df.columns else tr_df["raw_open"])).to_numpy(dtype=np.float64)
    model_high = (tr_df["tr_high"] if "tr_high" in tr_df.columns else (tr_df["model_high"] if "model_high" in tr_df.columns else tr_df["raw_high"])).to_numpy(dtype=np.float64)
    model_low = (tr_df["tr_low"] if "tr_low" in tr_df.columns else (tr_df["model_low"] if "model_low" in tr_df.columns else tr_df["raw_low"])).to_numpy(dtype=np.float64)
    model_close = (tr_df["tr_close"] if "tr_close" in tr_df.columns else (tr_df["model_close"] if "model_close" in tr_df.columns else tr_df["raw_close"])).to_numpy(dtype=np.float64)
    volume = (tr_df["raw_volume"] if "raw_volume" in tr_df.columns else (tr_df["volume"] if "volume" in tr_df.columns else tr_df["normalized_volume"])).to_numpy(dtype=np.float64)

    raw_features = np.column_stack([
        feats_df[c].to_numpy(dtype=np.float64) for c in EXPECTED_FEATURES_ORDERED
    ])

    target_63_legacy = labels_df["target_value"].to_numpy(dtype=np.float64)
    target_executable = np.full(T, np.nan, dtype=np.float64)

    bar_status = sched_val_df["bar_status"].to_numpy(dtype=str) if "bar_status" in sched_val_df.columns else np.array(["VALID"] * T)
    validity_flags = np.array([st == "VALID" for st in bar_status], dtype=bool)
    feat_valid_mask = feats_df["valid_mask"].to_numpy(dtype=bool) if "valid_mask" in feats_df.columns else validity_flags

    if "segment_id" in tr_df.columns:
        segment_ids = tr_df["segment_id"].to_numpy(dtype=np.int64)
    elif "segment_id" in sched_val_df.columns:
        segment_ids = sched_val_df["segment_id"].to_numpy(dtype=np.int64)
    elif "segment_id" in val_df.columns:
        segment_ids = val_df["segment_id"].to_numpy(dtype=np.int64)
    elif "segment_id" in df_aligned.columns:
        segment_ids = df_aligned["segment_id"].to_numpy(dtype=np.int64)
    else:
        segment_ids = np.zeros(T, dtype=np.int64)

    return SecurityFeatureCache(
        security_id=sec_id,
        sessions=sessions,
        raw_open=raw_open,
        raw_high=raw_high,
        raw_low=raw_low,
        raw_close=raw_close,
        model_open=model_open,
        model_high=model_high,
        model_low=model_low,
        model_close=model_close,
        volume=volume,
        raw_features=raw_features,
        target_63_legacy=target_63_legacy,
        target_executable=target_executable,
        session_ordinals=session_ordinals,
        validity_flags=validity_flags,
        bar_status=bar_status,
        file_sha256=file_sha,
        calendar_sha256=cal_sha,
        feature_valid_mask=feat_valid_mask,
        segment_ids=segment_ids,
    )


def get_or_build_security_cache(
    parquet_path: Union[str, Path],
    sec_cal: VenueCalendar,
    cache_dir: Union[str, Path],
    force_recompute: bool = False,
    mmap_mode: Optional[str] = "r",
) -> SecurityFeatureCache:
    """Retrieve existing cached security features (.npy + manifest) if valid, or compute and persist."""
    parquet_path = Path(parquet_path)
    cache_dir = Path(cache_dir)
    sec_dir = cache_dir / parquet_path.stem
    manifest_path = sec_dir / "manifest.json"

    file_bytes = None
    current_file_sha = None
    current_cal_sha = None

    def _get_hashes():
        nonlocal file_bytes, current_file_sha, current_cal_sha
        if current_file_sha is None:
            file_bytes = parquet_path.read_bytes()
            current_file_sha = hashlib.sha256(file_bytes).hexdigest()
            current_cal_sha = getattr(sec_cal, "schedule_hash", None) or hashlib.sha256("".join(sec_cal.sessions).encode()).hexdigest()
        return current_file_sha, current_cal_sha

    if manifest_path.exists() and not force_recompute:
        try:
            cache = SecurityFeatureCache.load(sec_dir, mmap_mode=mmap_mode)
            f_sha, c_sha = _get_hashes()
            if (cache.file_sha256 == f_sha and
                cache.calendar_sha256 == c_sha and
                cache.feature_spec_id == FEATURE_SPEC_ID and
                cache.target_spec_id == TARGET_SPEC_ID):
                return cache
        except Exception:
            pass  # Corrupted cache directory; recompute

    cache_file = cache_dir / f"{parquet_path.stem}.npz"
    if cache_file.exists() and not force_recompute:
        try:
            cache = SecurityFeatureCache.load(cache_file)
            f_sha, c_sha = _get_hashes()
            if (cache.file_sha256 == f_sha and
                cache.calendar_sha256 == c_sha and
                cache.feature_spec_id == FEATURE_SPEC_ID and
                cache.target_spec_id == TARGET_SPEC_ID):
                return cache
        except Exception:
            pass  # Corrupted cache file; recompute

    cache = compute_security_cache(parquet_path, sec_cal)
    cache.save(sec_dir)
    return cache
