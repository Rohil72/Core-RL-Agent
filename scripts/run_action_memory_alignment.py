"""Run the Memory-to-Action Alignment Study over frozen Phase 6 states and decision adapters."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.memory.calibration import fit_downside_calibration, DownsideCalibrationModel
from src.backtest.market_memory_backtester import PolicyConfig, run_long_only_backtest, compute_backtest_metrics
from src.backtest.market_memory_evaluator import run_market_memory_evaluation
from export_phase5_transfer_latents import run as export_transfer_latents
from run_phase5_decision_alignment import discover_adapter_sources, AdapterSource, _retrieval_frame


@dataclass(frozen=True)
class AlignmentSource:
    """One complete frozen encoder, adapter, memory, and market-data source."""

    source: AdapterSource
    adapter_checkpoint: Path
    train_decisions: Path
    market: str
    price_files: tuple[Path, ...]


def _file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def _payload_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_for_manifest(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _manifest_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _prepared_entry_is_valid(entry: dict[str, Any] | None) -> bool:
    """Validate prepared signal artifacts before allowing resume reuse."""

    if not entry:
        return False
    for prefix in ("development_signals", "temporal_signals"):
        path_value = entry.get(prefix)
        expected_hash = entry.get(f"{prefix}_sha256")
        if not path_value or not expected_hash:
            return False
        path = _manifest_path(str(path_value))
        if not path.is_file() or _file_hash(path) != expected_hash:
            return False
    return True


def _resolve_alignment_sources(config: dict[str, Any]) -> list[AlignmentSource]:
    """Resolve and validate every frozen artifact required by the real study."""

    experiment = config["experiment"]
    sources = discover_adapter_sources(config, project_root=PROJECT_ROOT)
    adapter_root = PROJECT_ROOT / experiment["adapter_run"]
    resolved: list[AlignmentSource] = []
    for source in sources:
        market = source.group.removeprefix("regional_")
        adapter_dir = adapter_root / source.group / f"seed_{source.seed}"
        adapter_checkpoint = adapter_dir / "decision_adapter.pt"
        train_decisions = adapter_dir / "train_decisions.parquet"
        required = {
            "encoder checkpoint": source.encoder_checkpoint,
            "adapter checkpoint": adapter_checkpoint,
            "adapter training decisions": train_decisions,
            "training latents": source.train_path,
            "validation latents": source.val_path,
        }
        missing = [label for label, path in required.items() if path is None or not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete frozen source {source.name}: missing {', '.join(missing)}.")
        price_files = tuple(Path(path) for path in sorted(glob.glob(str(PROJECT_ROOT / source.backtest_glob))))
        if not price_files:
            raise FileNotFoundError(f"No market files matched {source.backtest_glob!r} for {source.name}.")
        resolved.append(
            AlignmentSource(
                source=source,
                adapter_checkpoint=adapter_checkpoint,
                train_decisions=train_decisions,
                market=market,
                price_files=price_files,
            )
        )
    return resolved


def _source_hashes(sources: list[AlignmentSource]) -> dict[str, Any]:
    """Hash all immutable model, memory, latent, and price inputs."""

    return {
        item.source.name: {
            "encoder": _file_hash(Path(item.source.encoder_checkpoint)),
            "adapter": _file_hash(item.adapter_checkpoint),
            "train_decisions": _file_hash(item.train_decisions),
            "train_latents": _file_hash(item.source.train_path),
            "val_latents": _file_hash(item.source.val_path),
            "price_files": {
                _path_for_manifest(path): _file_hash(path)
                for path in item.price_files
            },
        }
        for item in sources
    }


def _memory_evaluation_config(
    base_config: dict[str, Any],
    source: AlignmentSource,
    train_decisions: Path,
    query_decisions: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build the established decision-space retrieval configuration."""

    memory = deepcopy(base_config["memory_defaults"])
    memory.update(
        {
            "target_upside": "decision_mfe",
            "target_alpha": "decision_net_alpha",
            "target_absolute_return": "decision_return_63",
            "target_downside": "decision_mae",
            "target_path_quality": "decision_path_quality",
            "target_holding_period": "decision_holding_sessions",
            "score_mode": "alpha_lcb",
            "require_outcome_availability": True,
            "same_ticker_mode": "exclude",
        }
    )
    return {
        "data": {
            "train_latents": str(train_decisions),
            "test_latents": str(query_decisions),
            "precomputed_glob": source.source.backtest_glob,
            "output_dir": str(output_dir),
        },
        "memory": memory,
        "policy": deepcopy(base_config["policy"]),
        "evaluation": {"baselines": "", "memory_metric_target": "decision_net_alpha"},
        "run": {"write_neighbors": True, "write_memory_reports": False},
    }


def _prepare_signal_period(
    source: AlignmentSource,
    base_config: dict[str, Any],
    export_config: Path,
    output_root: Path,
    period: str,
    start: str,
    end: str,
    *,
    resume: bool,
) -> Path:
    """Export frozen decisions and score their historical memory evidence."""

    source_root = output_root / "prepared_sources" / source.source.name
    decisions = source_root / "decisions" / f"{period}.parquet"
    retrieval_root = source_root / "retrieval" / period
    memory = retrieval_root / "train.parquet"
    query = retrieval_root / "query.parquet"
    evaluation_root = source_root / "memory" / period
    signals = evaluation_root / "eval" / "signals.parquet"
    neighbors = evaluation_root / "eval" / "neighbors.parquet"
    if resume and all(path.is_file() and path.stat().st_size > 0 for path in (decisions, memory, query, signals, neighbors)):
        return signals

    decisions.parent.mkdir(parents=True, exist_ok=True)
    export_transfer_latents(
        encoder_checkpoint=str(source.source.encoder_checkpoint),
        adapter_checkpoint=str(source.adapter_checkpoint),
        target_glob=source.source.backtest_glob,
        start=start,
        end=end,
        output=str(decisions),
        config=str(export_config),
    )
    _retrieval_frame(source.train_decisions, memory)
    _retrieval_frame(decisions, query)
    evaluation = _memory_evaluation_config(base_config, source, memory, query, evaluation_root)
    generated = source_root / "generated_configs" / f"memory_{period}.yaml"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(yaml.safe_dump(evaluation, sort_keys=False), encoding="utf-8")
    result = run_market_memory_evaluation(evaluation, PROJECT_ROOT, run_id="eval")
    result_signals = result / "signals.parquet"
    if result_signals != signals or not signals.is_file():
        raise RuntimeError(f"Memory scoring did not produce the expected signals for {source.source.name}/{period}.")
    return signals


def _prepare_real_signals(
    sources: list[AlignmentSource],
    base_config: dict[str, Any],
    adapter_config: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    *,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build development and temporal signals from frozen real-market artifacts."""

    export_values = _deep_merge(adapter_config, config.get("common", {}))
    export_config = output_root / "generated_configs" / "frozen_inference.yaml"
    export_config.parent.mkdir(parents=True, exist_ok=True)
    export_config.write_text(yaml.safe_dump(export_values, sort_keys=False), encoding="utf-8")
    calibration = config["common"]["calibration"]
    protocols = config["common"]["temporal_evaluation"]["periods"]
    temporal_start = min(str(period["start"]) for period in protocols.values())
    temporal_end = max(str(period.get("query_end", period["end"])) for period in protocols.values())
    development_frames: list[pd.DataFrame] = []
    temporal_frames: list[pd.DataFrame] = []
    manifest_path = output_root / "prepared_sources_manifest.json"
    existing_artifacts = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if resume and manifest_path.is_file()
        else {}
    )
    artifacts: dict[str, Any] = {}
    for source in sources:
        existing = existing_artifacts.get(source.source.name)
        if _prepared_entry_is_valid(existing):
            development_path = _manifest_path(existing["development_signals"])
            temporal_path = _manifest_path(existing["temporal_signals"])
        else:
            development_path = _prepare_signal_period(
                source,
                base_config,
                export_config,
                output_root,
                "development",
                str(calibration["start_date"]),
                str(calibration["end_date"]),
                resume=False,
            )
            temporal_path = _prepare_signal_period(
                source,
                base_config,
                export_config,
                output_root,
                "temporal",
                temporal_start,
                temporal_end,
                resume=False,
            )
        development = pd.read_parquet(development_path)
        temporal = pd.read_parquet(temporal_path)
        for frame in (development, temporal):
            frame["market"] = source.market
            frame["seed"] = source.source.seed
            frame["source_name"] = source.source.name
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        development_frames.append(development)
        temporal_frames.append(temporal)
        artifacts[source.source.name] = {
            "development_signals": _path_for_manifest(development_path),
            "development_signals_sha256": _file_hash(development_path),
            "temporal_signals": _path_for_manifest(temporal_path),
            "temporal_signals_sha256": _file_hash(temporal_path),
        }
        manifest_path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    return (
        pd.concat(development_frames, ignore_index=True),
        pd.concat(temporal_frames, ignore_index=True),
        artifacts,
    )


def _git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode("utf-8").strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT).decode("utf-8").strip()
        is_dirty = len(status) > 0
        return commit, is_dirty
    except Exception:
        return "unknown", True


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _compute_profit_concentration(trades: pd.DataFrame) -> float:
    if trades.empty or "pnl" not in trades:
        return 0.0
    pnls = trades["pnl"].to_numpy(dtype=float)
    pos_pnl = pnls[pnls > 0]
    if len(pos_pnl) == 0 or pos_pnl.sum() <= 0:
        return 0.0
    return float(pos_pnl.max() / pos_pnl.sum())


def create_synthetic_smoke_data() -> pd.DataFrame:
    """Create a minimal synthetic dataframe for offline smoke testing."""
    dates = pd.DatetimeIndex(
        [
            *pd.date_range("2024-11-01", periods=40, freq="B", tz="UTC"),
            *pd.date_range("2025-01-01", periods=80, freq="B", tz="UTC"),
            *pd.date_range("2026-01-01", "2026-06-30", freq="B", tz="UTC"),
        ]
    ).drop_duplicates()
    tickers = ["AAPL"]
    rows = []
    np.random.seed(42)
    price = 100.0
    for d in dates:
        for t in tickers:
            ret = np.random.normal(0.0005, 0.015)
            price = max(10.0, price * (1.0 + ret))
            alpha = np.random.normal(0.002, 0.01)
            downside = -abs(np.random.normal(0.03, 0.01))
            mae = downside * np.random.uniform(0.8, 1.5)
            mfe = abs(np.random.normal(0.04, 0.02))
            ret63 = np.random.normal(0.02, 0.05)
            rows.append(
                {
                    "timestamp": d,
                    "ticker": t,
                    "market": "US",
                    "open": price,
                    "close": price * (1.0 + np.random.normal(0, 0.002)),
                    "latent_0": np.random.normal(0, 1),
                    "latent_1": np.random.normal(0, 1),
                    "retrieval_expected_upside": mfe,
                    "retrieval_expected_alpha": alpha,
                    "retrieval_alpha_ci_low": alpha - 0.005,
                    "retrieval_alpha_lcb": alpha - 0.005,
                    "retrieval_expected_absolute_return": ret63,
                    "retrieval_absolute_return_ci_low": ret63 - 0.01,
                    "retrieval_absolute_return_lcb": ret63 - 0.01,
                    "retrieval_expected_downside": downside,
                    "retrieval_confidence": 0.8,
                    "retrieval_neighbor_count": 25,
                    "retrieval_ood_pass": True,
                    "opportunity_score": alpha,
                    "decision_mae": mae,
                    "decision_mfe": mfe,
                    "decision_net_alpha": alpha,
                    "decision_return_63": ret63,
                }
            )
    return pd.DataFrame(rows)


def run_alignment_study(
    config_path: str,
    run_id: str,
    *,
    selected_variants: set[str] | None = None,
    resume: bool = False,
    smoke_test: bool = False,
    preflight_only: bool = False,
    output_root_override: str | Path | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))

    # Load Base Config to merge policy defaults (Fixes P0 Finding #3)
    base_config_path = PROJECT_ROOT / config["study"]["base_config"]
    base_cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    adapter_config_path = PROJECT_ROOT / config["study"]["adapter_config"]
    adapter_cfg = yaml.safe_load(adapter_config_path.read_text(encoding="utf-8"))

    if output_root_override is not None:
        output_root = Path(output_root_override)
    else:
        output_root = PROJECT_ROOT / config["study"]["output_root"] / run_id

    # Check non-empty directory reuse
    if output_root.exists() and any(output_root.iterdir()) and not resume and not preflight_only:
        raise FileExistsError(
            f"Refusing to reuse non-empty run directory {output_root}. Pass --resume to reuse verified artifacts."
        )

    output_root.mkdir(parents=True, exist_ok=True)

    # Source preflight fails closed: no implicit fallback and no synthetic production path.
    study_merged_config = _deep_merge(adapter_cfg, config.get("common", {}))
    real_sources = [] if smoke_test else _resolve_alignment_sources(study_merged_config)
    expected_source_count = int(config["study"].get("expected_source_count", len(real_sources)))
    if not smoke_test and len(real_sources) != expected_source_count:
        raise RuntimeError(
            f"Expected {expected_source_count} complete frozen sources, resolved {len(real_sources)}."
        )
    source_hashes = {"synthetic_smoke": {"seed": 42}} if smoke_test else _source_hashes(real_sources)

    code_hashes = {
        "config": _file_hash(config_file),
        "base_config": _file_hash(base_config_path),
        "adapter_config": _file_hash(adapter_config_path),
        "runner": _file_hash(Path(__file__)),
        "calibration": _file_hash(PROJECT_ROOT / "src/memory/calibration.py"),
        "backtester": _file_hash(PROJECT_ROOT / "src/backtest/market_memory_backtester.py"),
        "exporter": _file_hash(PROJECT_ROOT / "scripts/export_phase5_transfer_latents.py"),
        "memory_evaluator": _file_hash(PROJECT_ROOT / "src/backtest/market_memory_evaluator.py"),
    }
    input_fingerprint = _payload_hash({"code_hashes": code_hashes, "source_hashes": source_hashes})

    if preflight_only:
        preflight_info = {
            "status": "preflight_passed",
            "source_count": len(real_sources) if not smoke_test else 1,
            "sources": [s.source.name for s in real_sources] if not smoke_test else ["synthetic_US_seed_7"],
            "source_hashes": source_hashes,
            "code_hashes": code_hashes,
            "input_fingerprint": input_fingerprint,
            "smoke_test": smoke_test,
        }
        (output_root / "source_preflight.json").write_text(json.dumps(preflight_info, indent=2), encoding="utf-8")
        return preflight_info

    git_commit, git_dirty = _git_info()
    command_line = " ".join(sys.argv)
    timestamp_start = datetime.now(timezone.utc).isoformat()

    # Resume verification covers every declared code and immutable data input.
    prov_file = output_root / "provenance.json"
    if resume and not prov_file.exists():
        raise RuntimeError("Cannot resume run: provenance.json is missing.")
    if resume:
        existing_prov = json.loads(prov_file.read_text(encoding="utf-8"))
        if existing_prov.get("input_fingerprint") != input_fingerprint:
            raise RuntimeError("Cannot resume run: code or immutable source hashes changed.")

    initial_provenance = {
        "run_id": run_id,
        "timestamp": timestamp_start,
        "command_line": command_line,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "code_hashes": code_hashes,
        "source_hashes": source_hashes,
        "input_fingerprint": input_fingerprint,
        "status": "running",
        "smoke_test": smoke_test,
    }
    prov_file.write_text(json.dumps(initial_provenance, indent=2), encoding="utf-8")

    variants_cfg = config["variants"]
    variant_names = [name for name in variants_cfg if not selected_variants or name in selected_variants]

    # Generate variant definitions markdown
    variant_defs_md = ["# Action Memory Alignment Study: Variant Definitions\n"]
    for name, v_info in variants_cfg.items():
        variant_defs_md.append(f"## Variant `{name}`")
        variant_defs_md.append(f"- **Description**: {v_info.get('description', '')}")
        p_info = v_info.get("policy", {})
        for pk, pv in p_info.items():
            variant_defs_md.append(f"  - `{pk}`: `{pv}`")
        variant_defs_md.append("")
    (output_root / "variant_definitions.md").write_text("\n".join(variant_defs_md), encoding="utf-8")

    # Build retrieval signals from the frozen stack. Synthetic data is smoke-only.
    if smoke_test:
        raw_signals = create_synthetic_smoke_data()
        dev_signals = raw_signals[
            (raw_signals["timestamp"] >= pd.Timestamp("2022-01-01", tz="UTC"))
            & (raw_signals["timestamp"] < pd.Timestamp("2025-01-01", tz="UTC"))
        ].copy()
        temporal_signals = raw_signals[
            raw_signals["timestamp"] >= pd.Timestamp("2025-01-01", tz="UTC")
        ].copy()
        prepared_artifacts: dict[str, Any] = {"synthetic_smoke": True}
        markets = ["US"]
        seeds = [7]
    else:
        dev_signals, temporal_signals, prepared_artifacts = _prepare_real_signals(
            real_sources,
            base_cfg,
            adapter_cfg,
            config,
            output_root,
            resume=resume,
        )
        markets = config["common"]["experiment"]["active_markets"]
        seeds = config["common"]["experiment"]["active_seeds"]

    # Downside calibration is fit exclusively on causally scored development signals.
    cal_cfg = config["common"].get("calibration", {})
    calibration_model = fit_downside_calibration(
        dev_signals,
        adverse_quantile=float(cal_cfg.get("adverse_quantile", 0.90)),
        min_samples=int(cal_cfg.get("min_samples", 30)),
        start_date=str(cal_cfg.get("start_date", "2022-01-01")),
        end_date=str(cal_cfg.get("end_date", "2024-12-31")),
        source_hash=_payload_hash(prepared_artifacts),
    )

    (output_root / "calibration_manifest.json").write_text(
        json.dumps(calibration_model.to_dict(), indent=2), encoding="utf-8"
    )

    calibrated_signals = calibration_model.apply(temporal_signals)

    # Accounting protocols (Fixes P0 Finding #2)
    protocols = config["common"].get("temporal_evaluation", {}).get("periods", {})
    if not protocols:
        protocols = {
            "calendar_portfolio": {
                "start": "2025-01-01",
                "end": "2026-03-31",
                "query_end": "2026-03-31",
                "entry_cutoff": "2025-12-31",
                "accounting_cutoff": "2026-03-31",
                "protocol": "calendar_portfolio",
            },
            "q1_to_q1_entry_cohort": {
                "start": "2025-01-01",
                "end": "2026-03-31",
                "query_end": "2026-06-30",
                "entry_cutoff": "2026-03-31",
                "accounting_cutoff": "2026-06-30",
                "protocol": "q1_to_q1_entry_cohort",
            },
        }

    paired_rows: list[dict] = []
    market_rows: list[dict] = []
    entry_rows: list[dict] = []
    portfolio_rows: list[dict] = []
    censoring_rows: list[dict] = []
    decision_log_rows: list[dict] = []

    # Policy Defaults from Base Config (Fixes P0 Finding #3)
    base_policy_defaults = base_cfg.get("policy", {})

    for protocol_name, p_spec in protocols.items():
        protocol_start = pd.Timestamp(str(p_spec["start"]), tz="UTC")
        protocol_end = pd.Timestamp(str(p_spec["end"]), tz="UTC")
        query_end = pd.Timestamp(str(p_spec.get("query_end", p_spec["end"])), tz="UTC")
        entry_cutoff = p_spec.get("entry_cutoff")
        accounting_cutoff = p_spec.get("accounting_cutoff")

        for variant_name in variant_names:
            v_cfg = variants_cfg[variant_name]
            v_policy = deepcopy(v_cfg.get("policy", {}))
            diagnostic_only = bool(v_policy.pop("diagnostic_only", False))
            promotion_allowed = bool(v_policy.pop("promotion_allowed", True))

            # Deep merge base policy defaults + variant overrides
            merged_policy_dict = _deep_merge(base_policy_defaults, v_policy)
            merged_policy_dict["entry_cutoff_date"] = entry_cutoff
            merged_policy_dict["accounting_cutoff_date"] = accounting_cutoff
            merged_policy_dict["accounting_protocol"] = protocol_name

            for market in markets:
                for seed in seeds:
                    m_signals = calibrated_signals[
                        (calibrated_signals["market"] == market)
                        & (calibrated_signals.get("seed", seed) == seed)
                        & (calibrated_signals["timestamp"] >= protocol_start)
                        & (calibrated_signals["timestamp"] <= query_end)
                    ].copy()

                    if m_signals.empty:
                        raise RuntimeError(
                            f"No bounded temporal signals for {market} seed {seed} under {protocol_name}."
                        )

                    policy = PolicyConfig(**merged_policy_dict)

                    d_log: list[dict] = []
                    trades, equity = run_long_only_backtest(
                        m_signals,
                        policy,
                        score_col="opportunity_score",
                        decision_log=d_log,
                    )

                    metrics = compute_backtest_metrics(trades, equity, policy.initial_capital)
                    profit_conc = _compute_profit_concentration(trades)

                    # Entry quality uses the exact 63-session outcome attached at signal time.
                    if not trades.empty:
                        accepted_trades = trades[~trades["is_rebalance"].fillna(False)].copy()
                        accepted_trades = accepted_trades.drop_duplicates("trade_id", keep="last")
                        mature = accepted_trades[
                            accepted_trades["outcome_mature"].fillna(False)
                            & accepted_trades["realized_return_63"].notna()
                        ]
                        entry_cnt = len(accepted_trades)
                        mature_cnt = len(mature)
                        hit_rt = float((mature["realized_return_63"] > 0).mean()) if mature_cnt else None
                        mean_ret = float(mature["realized_return_63"].mean()) if mature_cnt else None
                        mean_alpha = float(mature["realized_alpha_63"].mean()) if mature_cnt else None
                        sev_mae = float((mature["realized_mae_63"] <= -0.10).mean()) if mature_cnt else None
                    else:
                        entry_cnt = 0
                        mature_cnt = 0
                        hit_rt = None
                        mean_ret = None
                        mean_alpha = None
                        sev_mae = None

                    entry_rows.append(
                        {
                            "protocol": protocol_name,
                            "variant": variant_name,
                            "market": market,
                            "seed": seed,
                            "entry_count": entry_cnt,
                            "mature_entry_count": mature_cnt,
                            "censored_entry_count": entry_cnt - mature_cnt,
                            "hit_rate": hit_rt,
                            "mean_entry_return": mean_ret,
                            "mean_entry_alpha": mean_alpha,
                            "severe_mae_frequency": sev_mae,
                        }
                    )

                    # Record per-decision audit (Fixes P0 Finding #4)
                    for d_item in d_log:
                        d_item.update(
                            {
                                "protocol": protocol_name,
                                "variant": variant_name,
                                "market": market,
                                "seed": seed,
                            }
                        )
                        decision_log_rows.append(d_item)

                    entered_decisions = [item for item in d_log if item.get("decision") == "enter"]
                    mature_decisions = sum(bool(item.get("outcome_mature", False)) for item in entered_decisions)
                    censoring_rows.append(
                        {
                            "protocol": protocol_name,
                            "variant": variant_name,
                            "market": market,
                            "seed": seed,
                            "entry_cutoff": entry_cutoff or "2026-03-31",
                            "protocol_start": protocol_start.isoformat(),
                            "entry_period_end": protocol_end.isoformat(),
                            "query_end": query_end.isoformat(),
                            "accounting_cutoff": accounting_cutoff,
                            "total_entries": len(entered_decisions),
                            "mature_entries": mature_decisions,
                            "censored_entries": len(entered_decisions) - mature_decisions,
                            "forced_liquidations": int((trades["exit_reason"] == "end_of_test").sum()) if not trades.empty else 0,
                        }
                    )

                    portfolio_rows.append(
                        {
                            "protocol": protocol_name,
                            "variant": variant_name,
                            "market": market,
                            "seed": seed,
                            "sharpe": metrics["sharpe"],
                            "total_return": metrics["total_return"],
                            "max_drawdown": metrics["max_drawdown"],
                            "trade_count": metrics["trade_count"],
                            "win_rate": metrics["win_rate"],
                            "profit_concentration": profit_conc,
                            "diagnostic_only": diagnostic_only,
                            "promotion_allowed": promotion_allowed,
                        }
                    )

    expected_evaluations = len(protocols) * len(variant_names) * len(markets) * len(seeds)
    if len(portfolio_rows) != expected_evaluations:
        raise RuntimeError(
            f"Incomplete protocol pairing: expected {expected_evaluations}, produced {len(portfolio_rows)}."
        )

    # Save dataframes
    pd.DataFrame(portfolio_rows).to_csv(output_root / "portfolio_results.csv", index=False)
    pd.DataFrame(censoring_rows).to_csv(output_root / "censoring_audit.csv", index=False)
    pd.DataFrame(entry_rows).to_csv(output_root / "entry_quality.csv", index=False)
    pd.DataFrame(decision_log_rows).to_parquet(output_root / "decision_audit.parquet", index=False)

    # Calculate paired results against C0
    port_df = pd.DataFrame(portfolio_rows)
    primary_protocol = config.get("comparison", {}).get("metric_namespace", "calendar_portfolio")
    cal_port_df = port_df[port_df["protocol"] == primary_protocol]

    if "C0" in cal_port_df["variant"].unique():
        c0_df = cal_port_df[cal_port_df["variant"] == "C0"].set_index(["market", "seed"])
        for v in variant_names:
            if v == "C0":
                continue
            v_df = cal_port_df[cal_port_df["variant"] == v].set_index(["market", "seed"])
            for (m, s), row in v_df.iterrows():
                c0_row = c0_df.loc[(m, s)] if (m, s) in c0_df.index else None
                if c0_row is not None:
                    paired_rows.append(
                        {
                            "protocol": primary_protocol,
                            "candidate_variant": v,
                            "market": m,
                            "seed": s,
                            "c0_sharpe": c0_row["sharpe"],
                            "candidate_sharpe": row["sharpe"],
                            "sharpe_delta": row["sharpe"] - c0_row["sharpe"],
                            "sharpe_win": row["sharpe"] > c0_row["sharpe"],
                            "c0_drawdown": c0_row["max_drawdown"],
                            "candidate_drawdown": row["max_drawdown"],
                            "drawdown_win": row["max_drawdown"] >= c0_row["max_drawdown"],
                        }
                    )

    paired_df = pd.DataFrame(paired_rows) if paired_rows else pd.DataFrame(
        columns=["protocol", "candidate_variant", "market", "seed", "c0_sharpe", "candidate_sharpe", "sharpe_delta", "sharpe_win", "c0_drawdown", "candidate_drawdown", "drawdown_win"]
    )
    paired_df.to_csv(output_root / "paired_results.csv", index=False)

    # Market level summary
    market_summary = (
        cal_port_df.groupby(["variant", "market"], as_index=False)
        .agg(
            mean_sharpe=("sharpe", "mean"),
            mean_return=("total_return", "mean"),
            worst_drawdown=("max_drawdown", "min"),
            mean_trades=("trade_count", "mean"),
            win_rate=("win_rate", "mean"),
            profit_concentration=("profit_concentration", "mean"),
        )
    )
    market_summary.to_csv(output_root / "market_results.csv", index=False)

    # Gate evaluation for robustness verdict
    best_candidate = "C4" if "C4" in variant_names else variant_names[-1]
    c4_paired = paired_df[paired_df["candidate_variant"] == best_candidate] if not paired_df.empty else pd.DataFrame()

    paired_wins = int(c4_paired["sharpe_win"].sum()) if not c4_paired.empty else 0
    market_wins = int((c4_paired.groupby("market")["sharpe_delta"].mean() > 0).sum()) if not c4_paired.empty else 0

    verdict = {
        "status": "robustness_supported" if paired_wins >= 10 and market_wins >= 4 else "diagnostic_completed",
        "primary_protocol": primary_protocol,
        "best_candidate": best_candidate,
        "paired_run_count": len(c4_paired),
        "paired_run_wins": paired_wins,
        "market_wins": market_wins,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "smoke_test": smoke_test,
    }
    (output_root / "robustness_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    prepared_manifest = output_root / "prepared_sources_manifest.json"
    prepared_manifest.write_text(json.dumps(prepared_artifacts, indent=2), encoding="utf-8")
    output_hashes = {
        name: _file_hash(output_root / name)
        for name in (
            "calibration_manifest.json",
            "portfolio_results.csv",
            "censoring_audit.csv",
            "entry_quality.csv",
            "decision_audit.parquet",
            "paired_results.csv",
            "market_results.csv",
            "robustness_verdict.json",
            "prepared_sources_manifest.json",
        )
    }
    provenance = {
        **initial_provenance,
        "status": "completed",
        "input_fingerprint": input_fingerprint,
        "calibration_provenance": calibration_model.to_dict(),
        "prepared_artifacts": prepared_artifacts,
        "output_hashes": output_hashes,
        "variants": variant_names,
        "hardware_runtime": {"python": sys.version, "platform": sys.platform},
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    prov_md = [
        "# Provenance & Execution Audit\n",
        f"- **Run ID**: `{run_id}`",
        f"- **Timestamp**: `{timestamp_start}`",
        f"- **Git Commit**: `{git_commit}` (dirty: `{git_dirty}`)",
        f"- **Config Hash**: `{code_hashes['config']}`",
        f"- **Input Fingerprint**: `{input_fingerprint}`",
        f"- **Command**: `{command_line}`",
        f"- **Smoke Test**: `{smoke_test}`",
    ]
    (output_root / "provenance.md").write_text("\n".join(prov_md), encoding="utf-8")

    # DAG Manifest (Fixes P1 Finding #5)
    manifest = {
        "run_id": run_id,
        "dag_nodes": [
            {"id": "config", "type": "input", "file": str(config_file)},
            {"id": "base_config", "type": "input", "file": str(base_config_path)},
            {"id": "adapter_config", "type": "input", "file": str(adapter_config_path)},
            {"id": "sources", "type": "input", "source_count": len(real_sources) if not smoke_test else 1, "fingerprint": input_fingerprint},
            {"id": "frozen_inference", "type": "step", "inputs": ["adapter_config", "sources"], "output": "prepared_sources_manifest.json"},
            {"id": "memory_scoring", "type": "step", "inputs": ["base_config", "frozen_inference"], "outputs": ["prepared_sources/*/memory/development/eval/signals.parquet", "prepared_sources/*/memory/temporal/eval/signals.parquet"]},
            {"id": "calibration", "type": "step", "inputs": ["memory_scoring"], "output": "calibration_manifest.json"},
            {"id": "backtest_calendar", "type": "step", "inputs": ["calibration", "memory_scoring"], "output": "portfolio_results.csv"},
            {"id": "backtest_cohort", "type": "step", "inputs": ["calibration", "memory_scoring"], "output": "censoring_audit.csv"},
            {"id": "audit", "type": "step", "inputs": ["backtest_calendar", "backtest_cohort"], "outputs": ["decision_audit.parquet", "entry_quality.csv"]},
            {"id": "summary", "type": "step", "inputs": ["audit"], "outputs": ["robustness_verdict.json", "reviewer_summary.md"]},
        ],
        "commands": {
            "preflight": f"{sys.executable} scripts/run_action_memory_alignment.py --config {config_path} --run-id {run_id} --preflight",
            "execute": f"{sys.executable} scripts/run_action_memory_alignment.py --config {config_path} --run-id {run_id}",
            "resume": f"{sys.executable} scripts/run_action_memory_alignment.py --config {config_path} --run-id {run_id} --resume",
        },
    }
    (output_root / "experiment_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    # Reviewer Summary
    reviewer_md = [
        "# Reviewer Summary: Action Memory Alignment Study\n",
        f"**Run ID**: `{run_id}` | **Status**: `{verdict['status']}`\n",
        "## Key Performance Indicators",
        f"- **Primary Protocol**: `{primary_protocol}`",
        f"- **Paired Sharpe Wins**: `{paired_wins}/{len(c4_paired)}`",
        f"- **Market Wins**: `{market_wins}/{len(markets)}`",
        f"- **Development Downside Correction (90th percentile)**: `{calibration_model.pooled_correction:.4f}`",
        "\n## Variant Performance Summary\n",
        market_summary.to_markdown(index=False),
        "\n## Success Gate Verdict",
        f"- **Status**: `{verdict['status']}`",
        f"- **D1 Promotion**: Disallowed (Diagnostic Only)",
    ]
    (output_root / "reviewer_summary.md").write_text("\n".join(reviewer_md), encoding="utf-8")

    return {
        "run_id": run_id,
        "output_root": str(output_root),
        "variants": variant_names,
        "verdict": verdict,
        "smoke_test": smoke_test,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/action_memory_alignment.yaml")
    parser.add_argument("--run-id", default="action_memory_alignment_v1")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    res = run_alignment_study(
        args.config,
        args.run_id,
        selected_variants=set(args.variants) if args.variants else None,
        resume=args.resume,
        smoke_test=args.smoke_test,
        preflight_only=args.preflight,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
