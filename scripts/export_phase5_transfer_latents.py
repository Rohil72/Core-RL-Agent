"""Export one target market using only a frozen source encoder/standardizer/adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import ensure_sequence_model_features  # noqa: E402
from src.data.io_utils import read_dataframe  # noqa: E402
from src.data.sequence_dataset import CycleSequenceDataset, FeatureStandardizer  # noqa: E402
from src.decision.adapter import DecisionAdapter  # noqa: E402
from src.decision.dataset import DecisionDatasetConfig, build_decision_frame  # noqa: E402
from src.trainers.train_cycle_model import load_checkpoint  # noqa: E402


def _market_frame(paths: list[Path], feature_cols: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = read_dataframe(path)
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame["ticker"] = str(frame["ticker"].iloc[0]) if "ticker" in frame else path.stem
        frame = ensure_sequence_model_features(frame)
        if missing := [column for column in feature_cols if column not in frame]:
            raise ValueError(f"{path.name} lacks source features: {missing}")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No target-market parquet files matched.")
    return pd.concat(frames).sort_index()


def run(
    encoder_checkpoint: str,
    adapter_checkpoint: str,
    target_glob: str,
    start: str,
    end: str,
    output: str,
    config: str | None = None,
) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, payload = load_checkpoint(encoder_checkpoint, device)
    if hasattr(encoder, "enable_memory_update"):
        encoder.enable_memory_update = False
    adapter = DecisionAdapter.from_checkpoint(torch.load(adapter_checkpoint, map_location=device)).to(device).eval()
    feature_cols = list(payload["feature_cols"])
    target_cols = list(payload["future_target_cols"])
    paths = sorted(PROJECT_ROOT.glob(target_glob))
    raw = _market_frame(paths, feature_cols)
    standardizer = FeatureStandardizer(**payload["standardizer"])
    scaled = standardizer.transform(raw, feature_cols)
    dataset = CycleSequenceDataset(
        scaled, feature_cols, target_cols,
        window_size=int(payload["model_config"].get("window_size", 252)),
        target_start=pd.Timestamp(start, tz="UTC"), target_end=pd.Timestamp(end, tz="UTC"),
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    rows = []
    encoder.to(device).eval()
    with torch.no_grad():
        for batch in loader:
            state = encoder(batch["sequence"].to(device))
            decision = adapter(state["latent"])
            latent_values = state["latent"].cpu().numpy()
            decision_values = decision["decision"].cpu().numpy()
            quantiles = decision["utility_quantiles"].cpu().numpy()
            targets = batch["future_target"].numpy()
            for index in range(len(targets)):
                row = {"ticker": batch["ticker"][index], "timestamp": batch["timestamp"][index]}
                row.update({f"latent_{i}": float(value) for i, value in enumerate(latent_values[index])})
                row.update({f"decision_{i}": float(value) for i, value in enumerate(decision_values[index])})
                row.update({f"pred_utility_q{int(q * 100):02d}": float(quantiles[index, i]) for i, q in enumerate(adapter.config.quantiles)})
                row.update({name: float(targets[index, i]) for i, name in enumerate(target_cols)})
                rows.append(row)
    values = (
        yaml.safe_load(Path(config).read_text(encoding="utf-8")) if config else {}
    ) or {}
    dataset_config = DecisionDatasetConfig(**values.get("decision_dataset", {}))
    frame = build_decision_frame(pd.DataFrame(rows), paths, dataset_config)
    coverage_config = values.get("inference_coverage", {})
    window_size = int(payload["model_config"].get("window_size", 252))
    requested_start = pd.Timestamp(start, tz="UTC")
    requested_end = pd.Timestamp(end, tz="UTC")
    expected_by_ticker: dict[str, pd.DatetimeIndex] = {}
    actual_by_ticker: dict[str, pd.DatetimeIndex] = {}
    for ticker, group in raw.groupby("ticker"):
        ordered = group.sort_index()
        eligible = ordered.index[window_size - 1 :]
        expected_by_ticker[str(ticker)] = pd.DatetimeIndex(
            eligible[(eligible >= requested_start) & (eligible <= requested_end)]
        )
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        actual_by_ticker = {
            str(ticker): pd.DatetimeIndex(group["timestamp"].sort_values())
            for ticker, group in frame.groupby("ticker")
        }
    coverage_rows = []
    for ticker, expected in expected_by_ticker.items():
        actual = actual_by_ticker.get(ticker, pd.DatetimeIndex([], tz="UTC"))
        expected_set = set(expected.asi8)
        actual_set = set(actual.asi8)
        covered = len(expected_set.intersection(actual_set))
        missing_tail = (
            int((expected > actual.max()).sum())
            if len(expected) and len(actual)
            else int(len(expected))
        )
        coverage_rows.append(
            {
                "ticker": ticker,
                "expected_rows": int(len(expected)),
                "actual_rows": int(len(actual)),
                "covered_rows": int(covered),
                "coverage_fraction": (
                    float(covered / len(expected)) if len(expected) else 1.0
                ),
                "missing_tail_sessions": missing_tail,
                "prediction_end": (
                    actual.max().isoformat() if len(actual) else None
                ),
                "expected_end": (
                    expected.max().isoformat() if len(expected) else None
                ),
            }
        )
    expected_total = sum(row["expected_rows"] for row in coverage_rows)
    covered_total = sum(row["covered_rows"] for row in coverage_rows)
    total_coverage = (
        float(covered_total / expected_total) if expected_total else 1.0
    )
    minimum_ticker_coverage = min(
        (row["coverage_fraction"] for row in coverage_rows),
        default=1.0,
    )
    maximum_missing_tail = max(
        (row["missing_tail_sessions"] for row in coverage_rows),
        default=0,
    )
    embedding_columns = [
        column
        for column in frame
        if column.startswith(("latent_", "decision_"))
        and column.rsplit("_", 1)[-1].isdigit()
    ]
    finite_embeddings = bool(
        not embedding_columns
        or np.isfinite(frame[embedding_columns].to_numpy(dtype=float)).all()
    )
    minimum_required = float(
        coverage_config.get("minimum_ticker_coverage", 0.0)
    )
    maximum_tail_allowed = int(
        coverage_config.get("maximum_missing_tail_sessions", 10**9)
    )
    require_finite = bool(
        coverage_config.get("require_finite_embeddings", False)
    )
    incomplete = [
        row["ticker"]
        for row in coverage_rows
        if row["coverage_fraction"] < minimum_required
        or row["missing_tail_sessions"] > maximum_tail_allowed
    ]
    if incomplete:
        raise RuntimeError(
            "Inference coverage contract failed for "
            + ", ".join(incomplete)
            + f"; minimum coverage={minimum_required}, "
            + f"maximum tail gap={maximum_tail_allowed}."
        )
    if require_finite and not finite_embeddings:
        raise FloatingPointError("Export produced non-finite latent embeddings.")
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    summary = {
        "output": str(destination), "rows": len(frame), "tickers": int(frame["ticker"].nunique()),
        "source_normalization": str(encoder_checkpoint), "target_fit_performed": False,
        "period": [start, end],
        "decision_dataset": values.get("decision_dataset", {}),
        "coverage": {
            "expected_rows": int(expected_total),
            "covered_rows": int(covered_total),
            "total_fraction": total_coverage,
            "minimum_ticker_fraction": minimum_ticker_coverage,
            "maximum_missing_tail_sessions": maximum_missing_tail,
            "finite_embeddings": finite_embeddings,
            "tickers": coverage_rows,
        },
    }
    destination.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--target-glob", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
