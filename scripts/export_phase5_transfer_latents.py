"""Export one target market using only a frozen source encoder/standardizer/adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    summary = {
        "output": str(destination), "rows": len(frame), "tickers": int(frame["ticker"].nunique()),
        "source_normalization": str(encoder_checkpoint), "target_fit_performed": False,
        "period": [start, end],
        "decision_dataset": values.get("decision_dataset", {}),
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
