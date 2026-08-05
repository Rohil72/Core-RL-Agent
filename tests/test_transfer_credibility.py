from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import scripts.run_final_memory_study as study
from src.eval.tabular_decoders import DecoderSuiteConfig, fit_predict_frozen_decoders
from src.eval.transfer_credibility import (
    country_jackknife,
    paired_block_bootstrap,
    reliability_deciles,
)
from src.memory.market_memory import _evidence_prior_weights
from src.memory.retrieval import RetrievalConfig, build_retrieval_index, retrieve_neighbors


def test_retrieval_hard_age_window_is_causal():
    memory = pd.DataFrame(
        {
            "ticker": ["OLD", "NEW"],
            "timestamp": pd.to_datetime(["2015-01-01", "2023-01-01"], utc=True),
        }
    )
    matrix = np.asarray([[0.0], [0.1]], dtype=float)
    config = RetrievalConfig(
        k=2,
        minimum_neighbor_separation_sessions=0,
        maximum_memory_age_days=1461,
    )
    index = build_retrieval_index(memory, matrix, config)

    indices, _ = retrieve_neighbors(
        memory,
        matrix,
        np.asarray([0.0]),
        pd.Timestamp("2024-01-01", tz="UTC"),
        "QUERY",
        config,
        retrieval_index=index,
    )

    assert indices.tolist() == [1]


def test_recency_and_market_balance_priors_are_explicit():
    neighbors = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2023-01-01", "2022-01-01", "2022-01-01"], utc=True
            ),
            "market": ["A", "B", "B"],
        }
    )

    weights = _evidence_prior_weights(
        neighbors,
        pd.Timestamp("2024-01-01", tz="UTC"),
        temporal_half_life_days=365,
        balance_group_column="market",
    )

    assert np.allclose(weights, [0.5, 0.125, 0.125], atol=0.01)


def test_frozen_tabular_decoders_use_only_mature_training_rows():
    rng = np.random.default_rng(7)
    training = pd.DataFrame(
        {
            "ticker": ["A"] * 80,
            "timestamp": pd.date_range("2019-01-01", periods=80, freq="B", tz="UTC"),
            "outcome_available_timestamp": pd.date_range(
                "2019-04-01", periods=80, freq="B", tz="UTC"
            ),
        }
    )
    for index in range(4):
        training[f"latent_{index}"] = rng.normal(size=len(training))
    training["future_blended_alpha_63"] = (
        0.02 * training["latent_0"] - 0.01 * training["latent_1"]
    )
    query = training.iloc[:10].copy()
    query["timestamp"] = pd.date_range("2022-01-03", periods=10, freq="B", tz="UTC")

    predictions, audit = fit_predict_frozen_decoders(
        training,
        query,
        DecoderSuiteConfig(knn_neighbors=5, pca_components=2, tree_iterations=20),
    )

    assert sorted(predictions) == ["elastic_net", "hist_gradient_boosting", "pca_knn"]
    assert all(len(values) == len(query) for values in predictions.values())
    assert audit["training_rows"] == len(training)


def test_bootstrap_jackknife_and_reliability_outputs_are_reproducible():
    dates = pd.date_range("2022-01-03", periods=120, freq="B", tz="UTC")
    candidate = pd.Series(np.linspace(-0.002, 0.004, len(dates)), index=dates)
    baseline = pd.Series(np.linspace(-0.002, 0.002, len(dates)), index=dates)
    first = paired_block_bootstrap(
        candidate, baseline, block_length=21, samples=100, seed=7
    )
    second = paired_block_bootstrap(
        candidate, baseline, block_length=21, samples=100, seed=7
    )
    assert first == second
    assert first[0].observed_delta > 0.0

    jackknife = country_jackknife({"A": candidate, "B": baseline})
    assert set(jackknife["excluded_market"]) == {"A", "B"}

    signals = pd.DataFrame(
        {
            "retrieval_confidence": np.linspace(0.0, 1.0, 100),
            "future_blended_alpha_63": np.linspace(-0.02, 0.05, 100),
        }
    )
    deciles = reliability_deciles(signals)
    assert len(deciles) == 10
    assert deciles.iloc[-1]["mean_realized_alpha"] > deciles.iloc[0]["mean_realized_alpha"]


def test_final_transfer_manifest_is_frozen_durable_and_complete(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "PROJECT_ROOT", tmp_path)
    config = yaml.safe_load(
        Path("configs/final_transfer_credibility.yaml").read_text(encoding="utf-8")
    )
    config["experiment"]["output_root"] = "reports/transfer"
    testbed = yaml.safe_load(
        Path("configs/final_research_testbed.yaml").read_text(encoding="utf-8")
    )
    testbed_path = tmp_path / "testbed.yaml"
    testbed_path.write_text(yaml.safe_dump(testbed, sort_keys=False), encoding="utf-8")
    config["experiment"]["source_testbed_config"] = str(testbed_path)
    config_path = tmp_path / "transfer.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = study.build(str(config_path), "run", "python")
    manifest = yaml.safe_load(
        (tmp_path / "reports" / "transfer" / "run" / "experiment_manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    commands = [" ".join(job["command"]) for job in manifest["jobs"]]

    assert result["job_count"] == 418
    assert result["variant_count"] == 5
    assert result["gpu_job_count"] == 0
    assert not any(job["uses_gpu"] for job in manifest["jobs"])
    assert not any("train_cycle_model" in command for command in commands)
    assert not any("offline_policy" in command for command in commands)
    assert sum("run_transfer_credibility_audit.py" in command for command in commands) == 3
    assert sum("evaluate_frozen_tabular_decoders.py" in command for command in commands) == 18
    prepare = next(job for job in manifest["jobs"] if job["id"] == "prepare_global_US_seed_7")
    assert any(path.endswith("growing_raw.parquet") for path in prepare["expected_outputs"])


def test_comparison_control_is_resolved_by_contract_not_name():
    transfer = yaml.safe_load(
        Path("configs/final_transfer_credibility.yaml").read_text(encoding="utf-8")
    )
    repair = yaml.safe_load(
        Path("configs/final_memory_repair.yaml").read_text(encoding="utf-8")
    )

    assert study._comparison_control_id(transfer) == "m0_raw_static"
    assert study._comparison_control_id(repair) == "raw_c0_static"


def test_comparison_control_rejects_ambiguous_controls():
    config = {
        "variants": [
            {
                "id": "first",
                "embedding_space": "raw",
                "memory_mode": "static",
                "targets": "exact_c0",
            },
            {
                "id": "second",
                "embedding_space": "raw",
                "memory_mode": "static",
                "targets": "exact_c0",
            },
        ]
    }

    try:
        study._comparison_control_id(config)
    except RuntimeError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("Ambiguous comparison controls must fail closed.")
