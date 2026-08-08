from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.decision.adapter import DecisionAdapter, DecisionAdapterConfig
from src.decision.alignment import pcgrad_backward
from src.decision.dataset import DecisionDatasetConfig, build_decision_frame, temporal_event_class
from src.decision.losses import (
    DecisionLossConfig,
    decision_alignment_loss,
    environment_risk_variance,
    opportunity_allocation_losses,
    temporal_class_weights,
    temporal_event_probabilities,
)
from src.decision.trainer import DateGroupedBatchSampler, transform_decision_frame
from src.eval.confirmation_lock import create_confirmation_lock, mark_confirmation_executed
from src.models.patch_transformer_model import HierarchicalPatchTransformerCycleModel
from src.policy.offline_policy import OfflinePolicyDatasetConfig, build_offline_policy_dataset
import scripts.run_phase5_decision_alignment as alignment_runner
from scripts.run_phase5_decision_alignment import (
    AdapterSource,
    discover_adapter_sources,
    evaluate_aligned_decision_policies,
)
from src.backtest.market_memory_backtester import PolicyConfig


def test_decision_dataset_builds_causal_multi_horizon_utility(tmp_path):
    dates = pd.date_range("2020-01-01", periods=80, tz="UTC", freq="B")
    prices = pd.DataFrame({"timestamp": dates, "ticker": "AAA", "close": np.linspace(100.0, 140.0, len(dates))})
    price_path = tmp_path / "AAA.parquet"
    prices.to_parquet(price_path, index=False)
    latents = pd.DataFrame({"timestamp": dates[:10], "ticker": "AAA", "latent_0": 1.0})
    result = build_decision_frame(latents, [price_path], DecisionDatasetConfig())
    assert {"decision_return_1", "decision_return_63", "decision_mae", "decision_utility"}.issubset(result)
    assert result.loc[0, "decision_return_63"] > 0
    assert result.loc[0, "decision_outcome_available_timestamp"] == dates[63]
    assert result["decision_holding_sessions"].eq(63).all()
    assert result["decision_exit_reason"].eq("fixed_horizon").all()


def test_fixed_horizon_utility_does_not_change_when_a2_signals_are_present(tmp_path):
    dates = pd.date_range("2020-01-01", periods=80, tz="UTC", freq="B")
    prices = pd.DataFrame({"timestamp": dates, "ticker": "AAA", "close": np.linspace(100.0, 140.0, len(dates))})
    price_path = tmp_path / "AAA.parquet"
    prices.to_parquet(price_path, index=False)
    latents = pd.DataFrame({"timestamp": dates[:10], "ticker": "AAA", "latent_0": 1.0})
    signals = prices.assign(opportunity_score=np.linspace(1.0, -1.0, len(prices)))
    without_signals = build_decision_frame(latents, [price_path], DecisionDatasetConfig())
    with_signals = build_decision_frame(latents, [price_path], DecisionDatasetConfig(), signals)
    assert np.allclose(without_signals["decision_utility"], with_signals["decision_utility"], equal_nan=True)
    assert with_signals["decision_exit_reason"].eq("fixed_horizon").all()
    assert "decision_a2_return" in with_signals


def test_date_grouped_sampler_keeps_cross_sections_together():
    date_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    sampler = DateGroupedBatchSampler(date_ids, batch_size=5, shuffle=False, seed=7)
    batches = list(sampler)
    for date_id in torch.unique(date_ids):
        members = set(torch.where(date_ids == date_id)[0].tolist())
        assert any(members.issubset(set(batch)) for batch in batches)


def test_all_decision_losses_have_nonzero_gradients():
    model = DecisionAdapter(DecisionAdapterConfig(input_dim=8, hidden_dim=6, decision_dim=4, dropout=0.0))
    latent = torch.randn(12, 8)
    utility = torch.linspace(-0.2, 0.3, 12)
    outcomes = torch.randn(12, 7)
    dates = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    tickers = torch.tensor([0, 1, 2] * 4)
    total, parts = decision_alignment_loss(
        model(latent), utility, outcomes, dates, tickers, model.config.quantiles, DecisionLossConfig()
    )
    assert all(float(value.detach()) > 0 for value in parts.values())
    total.backward()
    assert model.projection[1].weight.grad is not None
    assert torch.isfinite(model.projection[1].weight.grad).all()


def test_temporal_event_class_preserves_event_identity_and_time_bucket():
    horizons = (5, 10, 21)
    upside_path = np.array([0.01, 0.03, 0.06, 0.08, 0.11])
    downside_path = np.array([-0.01, -0.04, -0.07, -0.11])
    assert temporal_event_class(upside_path, horizons, 0.10, 0.10) == (1, 5, "upside")
    assert temporal_event_class(downside_path, horizons, 0.10, 0.10) == (4, 4, "drawdown")
    assert temporal_event_class(np.zeros(21), horizons, 0.10, 0.10) == (0, 0, "none")


def test_temporal_head_has_ordered_quantiles_and_monotonic_event_risk():
    model = DecisionAdapter(
        DecisionAdapterConfig(
            input_dim=8,
            hidden_dim=6,
            decision_dim=4,
            dropout=0.0,
            temporal_horizons=(5, 10, 21),
            enforce_non_crossing_quantiles=True,
            include_cash_logit=True,
        )
    )
    output = model(torch.randn(12, 8))
    quantiles = output["utility_quantiles"]
    assert torch.all(quantiles[:, 0] <= quantiles[:, 1])
    assert torch.all(quantiles[:, 1] <= quantiles[:, 2])
    probabilities, cumulative = temporal_event_probabilities(output["event_logits"], 3)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(12))
    assert torch.all(torch.diff(cumulative, dim=1) >= -1e-7)


def test_temporal_class_weights_are_train_derived_and_capped():
    targets = torch.tensor([0] * 16 + [1] * 4 + [2])
    weights = temporal_class_weights(targets, 3, "inverse_sqrt", maximum_weight=2.0)

    assert weights is not None
    assert weights[0] < weights[1] < weights[2]
    assert float(weights.max()) <= 2.0


def test_action_export_preserves_cross_sectional_cash_identity():
    model = DecisionAdapter(
        DecisionAdapterConfig(input_dim=2, hidden_dim=4, decision_dim=3, dropout=0.0, include_cash_logit=True)
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 2, utc=True),
            "ticker": ["A", "B", "C", "A", "B"],
            "latent_0": [0.1, 0.2, 0.3, 0.4, 0.5],
            "latent_1": [0.5, 0.4, 0.3, 0.2, 0.1],
        }
    )
    transformed = transform_decision_frame(
        model,
        frame,
        torch.device("cpu"),
        loss_config=DecisionLossConfig(opportunity_temperature=0.10),
    )

    for _, cross_section in transformed.groupby("timestamp"):
        total = cross_section["pred_stock_allocation"].sum() + cross_section["pred_cash_probability"].iloc[0]
        assert np.isclose(total, 1.0)
        assert cross_section["pred_target_exposure"].nunique() == 1


def test_aligned_policy_evaluation_is_explicit_and_optional():
    rows = []
    for day in range(4):
        for ticker, score, allocation in (("A", 0.20, 0.60), ("B", 0.10, 0.20)):
            rows.append(
                {
                    "timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day),
                    "ticker": ticker,
                    "open": 100.0 + day,
                    "close": 100.0 + day,
                    "opportunity_score": score,
                    "retrieval_expected_upside": 0.10,
                    "retrieval_expected_downside": -0.02,
                    "retrieval_confidence": 1.0,
                    "pred_stock_allocation": allocation,
                    "pred_target_exposure": 0.80,
                    "pred_action_margin": 0.05 if ticker == "A" else -0.01,
                    "pred_upside_by_63": 0.60,
                    "pred_drawdown_by_63": 0.20,
                }
            )
    results = evaluate_aligned_decision_policies(
        pd.DataFrame(rows),
        PolicyConfig(top_k=1, min_hold_days=1, max_hold_days=2, slippage_bps=0),
    )

    assert set(results) == {"learned_action", "memory_gated"}
    assert results["learned_action"]["trade_count"] > 0
    assert evaluate_aligned_decision_policies(pd.DataFrame({"ticker": ["A"]}), PolicyConfig()) == {}


def test_pre_temporal_adapter_checkpoint_remains_loadable():
    original = DecisionAdapter(
        DecisionAdapterConfig(input_dim=8, hidden_dim=6, decision_dim=4, dropout=0.0)
    )
    payload = original.checkpoint()
    for key in ("temporal_horizons", "enforce_non_crossing_quantiles", "include_cash_logit"):
        payload["config"].pop(key)
    restored = DecisionAdapter.from_checkpoint(payload)
    latent = torch.randn(3, 8)
    assert torch.allclose(original(latent)["decision"], restored(latent)["decision"])


def test_phase6_source_discovery_uses_preserved_testbed_latents(tmp_path):
    latent_root = tmp_path / "reports" / "final_testbed" / "phase6_v1" / "latents"
    for name in ("regional_US_seed_7", "global_seed_7"):
        directory = latent_root / name
        directory.mkdir(parents=True)
        (directory / "train_latents.parquet").touch()
        (directory / "val_latents.parquet").touch()
    config = {
        "experiment": {
            "source_mode": "phase6_testbed",
            "source_testbed_run": "reports/final_testbed/phase6_v1",
            "data_root": "data/international",
            "source_representations": ["regional", "global"],
            "active_markets": ["US"],
            "active_seeds": [7],
        }
    }
    sources = discover_adapter_sources(config, project_root=tmp_path)
    assert [source.name for source in sources] == ["regional_US_seed_7", "global_seed_7"]
    assert sources[0].precomputed_globs == ("data/international/US/*.parquet",)
    assert sources[1].precomputed_globs == ("data/international/*/*.parquet",)
    assert discover_adapter_sources(config, max_runs=1, project_root=tmp_path)[0].name == "regional_US_seed_7"


def test_adapter_backtest_uses_decision_outcome_schema(tmp_path, monkeypatch):
    run_output = tmp_path / "adapter"
    run_output.mkdir()
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "timestamp": pd.to_datetime(["2021-01-01"], utc=True),
            "latent_0": [0.1],
            "decision_0": [0.2],
            "decision_mfe": [0.1],
            "decision_mae": [-0.03],
            "decision_net_alpha": [0.05],
            "decision_path_quality": [0.7],
            "decision_holding_sessions": [21],
        }
    )
    frame.to_parquet(run_output / "train_decisions.parquet", index=False)
    frame.to_parquet(run_output / "val_decisions.parquet", index=False)
    captured = {}

    def fake_evaluation(config, root, run_id):
        captured.update(config)
        destination = tmp_path / "evaluation" / run_id
        destination.mkdir(parents=True)
        (destination / "metrics.json").write_text('{"sharpe": 0.0}', encoding="utf-8")
        return destination

    monkeypatch.setattr(alignment_runner, "run_market_memory_evaluation", fake_evaluation)
    source = AdapterSource(
        name="regional_US_seed_7",
        group="regional_US",
        seed=7,
        train_path=tmp_path / "train.parquet",
        val_path=tmp_path / "val.parquet",
        precomputed_globs=("data/international/US/*.parquet",),
        backtest_glob="data/international/US/*.parquet",
    )
    result = alignment_runner._adapter_backtest(
        {"experiment": {"source_phase4e_run": "missing"}}, run_output, source
    )
    assert result["sharpe"] == 0.0
    assert captured["memory"]["target_alpha"] == "decision_net_alpha"
    assert captured["evaluation"]["memory_metric_target"] == "decision_net_alpha"


def test_opportunity_loss_penalizes_missing_positive_cross_section():
    scores = torch.tensor([-0.2, -0.1, -0.3], requires_grad=True)
    utility = torch.tensor([0.15, -0.04, 0.08])
    dates = torch.zeros(3, dtype=torch.long)
    allocation, coverage, regret = opportunity_allocation_losses(
        scores,
        utility,
        dates,
        torch.tensor(0.5, requires_grad=True),
        top_k=3,
        temperature=0.1,
        minimum_positive_utility=0.0,
    )
    assert float(allocation) > 0
    assert float(coverage) > 0
    assert float(regret) > 0
    (allocation + coverage).backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_environment_risk_variance_is_finite_with_trimmed_outlier():
    losses = torch.tensor([1.0, 1.1, 1.2, 100.0, 2.0, 2.1, 2.2, 200.0])
    environments = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    untrimmed = environment_risk_variance(
        losses, environments, trim_fraction=0.0, minimum_rows=2
    )
    trimmed = environment_risk_variance(
        losses, environments, trim_fraction=0.25, minimum_rows=2
    )
    assert torch.isfinite(trimmed)
    assert float(trimmed) < float(untrimmed)


def test_full_temporal_decision_loss_backpropagates():
    model = DecisionAdapter(
        DecisionAdapterConfig(
            input_dim=8,
            hidden_dim=6,
            decision_dim=4,
            dropout=0.0,
            temporal_horizons=(5, 10, 21),
            enforce_non_crossing_quantiles=True,
            include_cash_logit=True,
        )
    )
    latent = torch.randn(18, 8)
    utility = torch.linspace(-0.2, 0.3, 18)
    outcomes = torch.randn(18, 7)
    dates = torch.tensor([0] * 6 + [1] * 6 + [2] * 6)
    tickers = torch.tensor([0, 1, 2, 3, 4, 5] * 3)
    event_target = torch.tensor([0, 1, 2, 3, 4, 5] * 3)
    environments = torch.tensor([0, 0, 0, 1, 1, 1] * 3)
    config = DecisionLossConfig(
        temporal_event_weight=0.25,
        temporal_ranking_weight=0.05,
        temporal_coherence_weight=0.05,
        opportunity_weight=0.10,
        coverage_weight=0.25,
        environment_weight=0.10,
        environment_minimum_rows=3,
    )
    total, parts = decision_alignment_loss(
        model(latent),
        utility,
        outcomes,
        dates,
        tickers,
        model.config.quantiles,
        config,
        temporal_target=event_target,
        environment_ids=environments,
    )
    assert {"temporal_event", "opportunity", "coverage", "environment"}.issubset(parts)
    assert torch.isfinite(total)
    total.backward()
    assert model.event_head.weight.grad is not None
    assert torch.isfinite(model.event_head.weight.grad).all()


def test_static_memory_is_repeatable_and_differentiable():
    model = HierarchicalPatchTransformerCycleModel(
        input_dim=5, future_target_dim=3, d_model=8, nhead=2, num_layers=2,
        patch_size=2, latent_dim=6, num_memory_slots=3, dropout=0.0,
        memory_mode="static_parameter",
    )
    model.eval()
    sequence = torch.randn(4, 5, 12)
    first = model(sequence)["latent"]
    second = model(sequence)["latent"]
    assert torch.allclose(first, second)
    first.sum().backward()
    assert model.memory_slots.grad is not None


def test_pcgrad_produces_finite_combined_gradient():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    first = parameter.sum()
    second = -parameter[0] + parameter[1]
    pcgrad_backward((first, second), (parameter,))
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()


def test_offline_policies_share_one_dataset_contract():
    rows = []
    for day in pd.date_range("2023-01-01", periods=4, tz="UTC"):
        for ticker, score in (("A", 0.2), ("B", 0.1), ("C", -0.1)):
            rows.append({
                "timestamp": day, "ticker": ticker, "decision_0": score,
                "pred_utility_q50": score, "decision_net_alpha": score / 10,
            })
    dataset = build_offline_policy_dataset(pd.DataFrame(rows), OfflinePolicyDatasetConfig(top_k=2))
    assert dataset["actions"].shape == (20, 1)
    assert dataset["observations"].shape[0] == dataset["rewards"].shape[0]
    assert set(np.unique(dataset["actions"])) == {0.0, 0.25, 0.5, 0.75, 1.0}


def test_confirmation_lock_is_one_shot(tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"frozen")
    lock_path = tmp_path / "confirmation.json"
    create_confirmation_lock(lock_path, [artifact], ["US", "India"])
    result = tmp_path / "metrics.json"
    result.write_text("{}", encoding="utf-8")
    mark_confirmation_executed(lock_path, [result])
    with pytest.raises(FileExistsError):
        mark_confirmation_executed(lock_path, [result])


def test_unexecuted_confirmation_lock_rejects_changed_inputs(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    lock_path = tmp_path / "confirmation_lock.json"

    create_confirmation_lock(lock_path, [first], ["US"])

    with pytest.raises(FileExistsError, match="different immutable inputs"):
        create_confirmation_lock(lock_path, [second], ["US"])
