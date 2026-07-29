import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import scripts.run_final_memory_study as study
from scripts.export_phase5_transfer_latents import _sanitize_inference_features
from src.data.sequence_dataset import FeatureStandardizer
from src.decision.dataset import bounded_path_quality
from src.memory.market_memory import MarketMemoryConfig, score_market_memory
from src.memory.retrieval import cap_ticker_concentration


def _decision_frame() -> pd.DataFrame:
    dates = pd.date_range("2021-01-04", periods=3, freq="B", tz="UTC")
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 3,
            "timestamp": dates,
            "decision_outcome_available_timestamp": dates + pd.offsets.BDay(63),
            "decision_return_63": [0.10, 0.05, -0.02],
            "decision_benchmark_return": [0.03, 0.02, 0.01],
            "future_max_return_63": [0.12, 0.07, 0.01],
            "future_min_return_63": [-0.03, -0.02, -0.08],
            "event_upside_before_drawdown_126": [1.0, 1.0, 0.0],
            "event_peak_offset_63": [20, 30, 10],
            "decision_mfe": [0.12, 0.07, 0.01],
            "decision_mae": [-0.03, -0.02, -0.08],
            "decision_path_quality": [4.0, 3.5, 0.125],
            "decision_holding_sessions": [63, 63, 63],
        }
    )
    for index in range(4):
        frame[f"latent_{index}"] = index + np.arange(3)
    for index in range(2):
        frame[f"decision_{index}"] = index + 0.1 * np.arange(3)
    return frame


def test_retrieval_view_never_silently_uses_the_wrong_embedding():
    source = _decision_frame()

    raw = study.materialize_retrieval_view(source, "raw")
    adapter = study.materialize_retrieval_view(source, "adapter")

    assert study._embedding_columns(raw, "latent_") == [
        "latent_0",
        "latent_1",
        "latent_2",
        "latent_3",
    ]
    assert study._embedding_columns(adapter, "latent_") == [
        "latent_0",
        "latent_1",
    ]
    assert not study._embedding_columns(raw, "decision_")
    assert not study._embedding_columns(adapter, "decision_")
    assert adapter["retrieval_embedding_space"].eq("adapter").all()
    assert adapter["alpha_target_source"].eq(
        "derived_local_cross_sectional_alpha"
    ).all()
    assert np.allclose(adapter["future_blended_alpha_63"], [0.07, 0.03, -0.03])
    assert adapter["outcome_available_timestamp"].notna().all()


def test_bounded_path_quality_cannot_dominate_the_score():
    result = bounded_path_quality(
        np.asarray([0.12, 0.07, 0.01, np.nan]),
        np.asarray([-0.03, -0.02, -0.08, np.nan]),
    )

    assert np.all((result >= 0.0) & (result <= 1.0))
    assert np.allclose(result[:3], [0.8, 7 / 9, 1 / 9], atol=1e-4)
    assert result[3] == 0.0

    view = study.materialize_retrieval_view(
        _decision_frame(),
        "adapter",
        "bounded_excursion_share",
    )
    assert view["decision_path_quality_bounded"].between(0.0, 1.0).all()
    assert view["path_quality_target_source"].eq(
        "bounded_excursion_share"
    ).all()


def test_transfer_sanitization_is_causal_finite_and_audited():
    frame = pd.DataFrame(
        {
            "feature_a": [1.0, np.nan, np.inf, 1000.0],
            "feature_b": [2.0, 4.0, 6.0, 8.0],
        }
    )
    standardizer = FeatureStandardizer(
        mean={"feature_a": 1.0, "feature_b": 2.0},
        std={"feature_a": 2.0, "feature_b": 2.0},
    )

    scaled, audit = _sanitize_inference_features(
        frame,
        ["feature_a", "feature_b"],
        standardizer,
        maximum_absolute_zscore=25.0,
        maximum_repaired_fraction=0.30,
    )

    assert np.isfinite(scaled.to_numpy()).all()
    assert scaled["feature_a"].tolist() == [0.0, 0.0, 0.0, 25.0]
    assert scaled["feature_b"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert audit["repaired_cells"] == 2
    assert audit["clipped_cells"] == 1
    assert audit["method"] == "source_training_mean"


def test_transfer_sanitization_rejects_excessive_repairs():
    frame = pd.DataFrame({"feature": [np.nan, np.inf, 1.0]})
    standardizer = FeatureStandardizer(
        mean={"feature": 0.0},
        std={"feature": 1.0},
    )

    try:
        _sanitize_inference_features(
            frame,
            ["feature"],
            standardizer,
            maximum_absolute_zscore=25.0,
            maximum_repaired_fraction=0.50,
        )
    except RuntimeError as exc:
        assert "exceeded its configured limit" in str(exc)
    else:
        raise AssertionError("Excessive feature repair was not rejected.")


def test_ticker_concentration_cap_preserves_order():
    indices = np.arange(8)
    tickers = np.asarray(["A", "A", "A", "B", "B", "C", "C", "D"])

    result = cap_ticker_concentration(indices, tickers, maximum_per_ticker=2)

    assert result.tolist() == [0, 1, 3, 4, 5, 6, 7]


def test_multiscale_memory_emits_stability_and_ticker_balance():
    rows = []
    available = pd.Timestamp("2020-12-31", tz="UTC")
    for ticker_index, ticker in enumerate(("A", "B", "C", "D", "E", "F")):
        for observation in range(3):
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": pd.Timestamp(
                        f"2020-{ticker_index + 1:02d}-{observation + 1:02d}",
                        tz="UTC",
                    ),
                    "outcome_available_timestamp": available,
                    "latent_0": ticker_index * 0.1 + observation * 0.01,
                    "latent_1": observation * 0.05,
                    "future_max_return_63": 0.04 + ticker_index * 0.005,
                    "alpha": 0.01 + ticker_index * 0.002,
                    "future_min_return_63": -0.03,
                    "event_upside_before_drawdown_126": 1.0,
                    "event_peak_offset_63": 20,
                }
            )
    memory = pd.DataFrame(rows)
    query = pd.DataFrame(
        {
            "ticker": ["Q"],
            "timestamp": [pd.Timestamp("2021-06-01", tz="UTC")],
            "latent_0": [0.2],
            "latent_1": [0.05],
        }
    )

    signals, neighbors = score_market_memory(
        memory,
        query,
        MarketMemoryConfig(
            k=10,
            min_neighbors=3,
            minimum_neighbor_separation_sessions=0,
            same_ticker_mode="exclude",
            same_ticker_neighbor_limit=None,
            max_neighbors_per_ticker=2,
            target_alpha="alpha",
            score_mode="alpha_predictive_tail",
            neighbor_scales=(3, 5, 10),
            require_outcome_availability=True,
        ),
    )

    assert signals.loc[0, "retrieval_scale_count"] == 3
    assert signals.loc[0, "retrieval_neighbor_count"] == 10
    assert 0.0 <= signals.loc[0, "retrieval_scale_sign_agreement"] <= 1.0
    assert neighbors.groupby("neighbor_ticker").size().max() <= 2


def test_final_memory_manifest_is_cpu_only(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "PROJECT_ROOT", tmp_path)
    config = yaml.safe_load(Path("configs/final_memory_study.yaml").read_text())
    config["experiment"]["output_root"] = "reports/study"
    testbed = yaml.safe_load(Path("configs/final_research_testbed.yaml").read_text())
    testbed_path = tmp_path / "testbed.yaml"
    testbed_path.write_text(yaml.safe_dump(testbed, sort_keys=False), encoding="utf-8")
    config["experiment"]["source_testbed_config"] = str(testbed_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = study.build(str(config_path), "run", "python")
    manifest = yaml.safe_load(
        (tmp_path / "reports" / "study" / "run" / "experiment_manifest.yaml").read_text()
    )
    commands = [" ".join(job["command"]) for job in manifest["jobs"]]

    assert result["job_count"] == 322
    assert result["variant_count"] == 4
    assert result["gpu_job_count"] == 0
    assert not any(job["uses_gpu"] for job in manifest["jobs"])
    assert not any("train_cycle_model" in command for command in commands)
    assert not any("train_phase5_adapter" in command for command in commands)
    assert not any("offline_policy" in command for command in commands)
    assert json.loads(
        (tmp_path / "reports" / "study" / "run" / "build_summary.json").read_text()
    )["promotion_allowed"] is False


def test_final_repair_manifest_refreshes_global_inference_once_per_seed(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(study, "PROJECT_ROOT", tmp_path)
    config = yaml.safe_load(Path("configs/final_memory_repair.yaml").read_text())
    config["experiment"]["output_root"] = "reports/repair"
    testbed = yaml.safe_load(
        Path("configs/final_research_testbed.yaml").read_text()
    )
    testbed_path = tmp_path / "testbed.yaml"
    testbed_path.write_text(
        yaml.safe_dump(testbed, sort_keys=False),
        encoding="utf-8",
    )
    config["experiment"]["source_testbed_config"] = str(testbed_path)
    config_path = tmp_path / "repair.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    result = study.build(str(config_path), "run", "python")
    manifest = yaml.safe_load(
        (
            tmp_path
            / "reports"
            / "repair"
            / "run"
            / "experiment_manifest.yaml"
        ).read_text()
    )
    export_jobs = [
        job for job in manifest["jobs"] if job["id"].startswith("export_")
    ]

    assert result["job_count"] == 265
    assert result["variant_count"] == 3
    assert result["gpu_job_count"] == 18
    assert result["inference_refresh_jobs"] == 18
    assert len(export_jobs) == 18
    assert all(job["uses_gpu"] for job in export_jobs)
    assert all("global_" in job["id"] for job in export_jobs)
    assert not any(
        "train_cycle_model" in " ".join(job["command"])
        for job in manifest["jobs"]
    )


def test_aggregate_reports_baseline_lift_and_memory_diagnostics(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(study, "PROJECT_ROOT", tmp_path)
    config = yaml.safe_load(Path("configs/final_memory_study.yaml").read_text())
    config["experiment"]["output_root"] = "reports/study"
    config["experiment"]["markets"] = ["US"]
    config["gates"]["minimum_positive_markets"] = 1
    config["gates"]["maximum_profit_concentration"] = 1.0
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    root = (
        tmp_path
        / "reports"
        / "study"
        / "run"
        / "evaluation"
        / "adapter_c0_static"
        / "development"
        / "US"
    )
    root.mkdir(parents=True)
    payload = {
        "ensemble": {
            "total_return": 0.10,
            "sharpe": 1.5,
            "max_drawdown": -0.10,
        },
        "ensemble_sharpe_lift_vs_mean_seed": 0.20,
        "baselines": {
            "equal_weight_buy_hold": {"sharpe": 0.7},
            "momentum_21": {"sharpe": 0.5},
        },
        "diagnostics": {
            "mean_retrieval_confidence": 0.6,
            "mean_retrieval_agreement_score": 0.7,
            "mean_retrieval_historical_diversity": 0.8,
            "mean_retrieval_effective_sample_size": 12.0,
            "mean_retrieval_cross_ticker_rate": 1.0,
            "mean_retrieval_scale_score_std": 0.01,
            "mean_retrieval_scale_sign_agreement": 0.9,
        },
    }
    (root / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    equity_root = root / "baselines" / "local_rank_memory"
    equity_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-01-03", periods=4, freq="B", tz="UTC"),
            "equity": [100_000.0, 101_000.0, 102_000.0, 110_000.0],
        }
    ).to_csv(equity_root / "equity_curve.csv", index=False)

    result = study.aggregate_variant(
        str(config_path),
        "run",
        "adapter_c0_static",
        "development",
    )
    table = pd.read_csv(
        tmp_path
        / "reports"
        / "study"
        / "run"
        / "aggregate"
        / "adapter_c0_static"
        / "development"
        / "market_results.csv"
    )

    assert result["median_excess_sharpe_vs_equal_weight"] == 0.8
    assert result["median_excess_sharpe_vs_momentum"] == 1.0
    assert table.loc[0, "mean_scale_sign_agreement"] == 0.9
