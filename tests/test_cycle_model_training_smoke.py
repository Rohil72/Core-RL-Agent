from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.io_utils import write_dataframe
from src.trainers.train_cycle_model import train


def _make_ticker_frame(ticker: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2018-01-01", "2021-12-31", freq="B", tz="UTC")
    base = np.linspace(50, 140, len(index))
    wave = 8 * np.sin(np.linspace(0, 16 * np.pi, len(index)))
    noise = rng.normal(0, 1.0, len(index))
    close = np.maximum(base + wave + noise, 5.0)
    volume = 1_000_000 + 100_000 * np.sin(np.linspace(0, 8 * np.pi, len(index)))

    frame = pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": volume,
            "tech_return_1": pd.Series(close).pct_change().fillna(0.0).to_numpy(),
            "tech_volume_ratio": 1.0
            + 0.2 * np.sin(np.linspace(0, 6 * np.pi, len(index))),
            "tech_minervini_gate": (
                np.sin(np.linspace(0, 10 * np.pi, len(index))) > -0.2
            ).astype(float),
            "fund_minervini_score": np.clip(
                0.5 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, len(index))), 0.0, 1.0
            ),
            "fund_report_available": 1.0,
            "fund_days_since_report": np.mod(np.arange(len(index)), 63).astype(float),
            "ticker": ticker,
        },
        index=index,
    )
    return frame


def test_cycle_model_training_smoke(tmp_path: Path):
    data_dir = tmp_path / "precomputed"
    data_dir.mkdir()

    write_dataframe(_make_ticker_frame("AAA", 1), data_dir / "AAA.parquet")
    write_dataframe(_make_ticker_frame("BBB", 2), data_dir / "BBB.parquet")

    config = {
        "data": {"precomputed_dir": str(data_dir / "*.parquet")},
        "oracle": {
            "min_duration_days": 10,
            "max_duration_days": 63,
            "min_return": 0.10,
            "catastrophic_return": -0.10,
        },
        "model": {
            "window_size": 21,
            "encoder": "patch_transformer",
            "d_model": 32,
            "nhead": 4,
            "num_layers": 2,
            "patch_size": 5,
            "latent_dim": 32,
            "num_memory_slots": 8,
            "dropout": 0.0,
        },
        "training": {
            "seed": 7,
            "device": "cpu",
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "action_loss_weight": 0.75,
            "future_loss_weight": 0.2,
            "hard_negative_weight": 0.5,
            "grad_clip": 1.0,
            "model_dir": str(tmp_path / "models"),
            "auto_resume": True,
            "checkpoint_interval_steps": 100000,
        },
        "self_supervised": {
            "reconstruction_loss_weight": 0.05,
            "mask_probability": 0.10,
            "mask_span_probability": 0.05,
            "mask_span_length": 3,
            "mask_value": 0.0,
        },
        "split": {
            "train_years": 1,
            "val_years": 1,
            "test_years": 1,
            "step_years": 1,
            "selected_fold": -1,
            "ticker_holdout_fraction": 0.0,
            "seed": 7,
        },
        "features": {
            "sequence": [
                "tech_return_1",
                "tech_volume_ratio",
                "tech_minervini_gate",
                "fund_minervini_score",
                "fund_report_available",
                "fund_days_since_report",
            ],
            "future_targets": [
                "future_return_21",
                "future_return_63",
                "future_max_return_63",
                "future_min_return_63",
                "event_peak_offset_63",
                "event_drawdown_offset_63",
                "event_upside_before_drawdown_126",
            ],
        },
        "latent_export": {"splits": ["train", "val", "test"]},
        "evaluation": {
            "cooldown_days": 42,
            "catastrophic_return": -0.10,
            "late_exit_penalty_factor": 1.5,
            "reports_dir": str(tmp_path / "reports"),
        },
    }
    config_path = tmp_path / "cycle_model.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    result = train(str(config_path))

    assert Path(result["checkpoint_path"]).exists()
    assert Path(result["report_json"]).exists()
    assert Path(result["report_md"]).exists()
    assert "test" in result["metrics"]
    assert {"train", "val", "test"}.issubset(set(result["latent_exports"]))
    for path in result["latent_exports"].values():
        assert Path(path).exists()
    assert (tmp_path / "models" / "latest_training_state.pt").exists()
    assert (tmp_path / "models" / "training_complete.json").exists()

    resumed = train(str(config_path), resume=True)
    assert resumed["checkpoint_path"] == result["checkpoint_path"]
