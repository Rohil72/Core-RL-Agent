from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.decision.adapter import DecisionAdapter, DecisionAdapterConfig
from src.decision.alignment import pcgrad_backward
from src.decision.dataset import DecisionDatasetConfig, build_decision_frame
from src.decision.losses import DecisionLossConfig, decision_alignment_loss
from src.decision.trainer import DateGroupedBatchSampler
from src.eval.confirmation_lock import create_confirmation_lock, mark_confirmation_executed
from src.models.patch_transformer_model import HierarchicalPatchTransformerCycleModel
from src.policy.offline_policy import OfflinePolicyDatasetConfig, build_offline_policy_dataset


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
