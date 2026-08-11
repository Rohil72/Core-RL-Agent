import torch

from src.models import build_cycle_model
from src.models.cycle_reasoning_model import CycleReasoningModel
from src.models.patch_transformer_model import HierarchicalPatchTransformerCycleModel


def _assert_output_contract(model: torch.nn.Module, latent_dim: int) -> None:
    sequence = torch.randn(3, 6, 21)
    outputs = model(sequence)

    assert set(outputs) == {"latent", "future_pred", "action_logits"}
    assert outputs["latent"].shape == (3, latent_dim)
    assert outputs["future_pred"].shape == (3, 4)
    assert outputs["action_logits"].shape == (3, 4)


def test_build_cycle_model_keeps_temporal_cnn_available():
    model = build_cycle_model(
        model_cfg={
            "encoder": "temporal_cnn",
            "window_sizes": [5, 10, 21],
            "branch_hidden_dim": 8,
            "latent_dim": 16,
        },
        input_dim=6,
        future_target_dim=4,
    )

    assert isinstance(model, CycleReasoningModel)
    _assert_output_contract(model, latent_dim=16)


def test_build_cycle_model_supports_patch_transformer():
    model = build_cycle_model(
        model_cfg={
            "encoder": "patch_transformer",
            "d_model": 32,
            "nhead": 4,
            "num_layers": 2,
            "patch_size": 5,
            "latent_dim": 16,
            "dropout": 0.0,
        },
        input_dim=6,
        future_target_dim=4,
    )

    assert isinstance(model, HierarchicalPatchTransformerCycleModel)
    _assert_output_contract(model, latent_dim=16)

    sequence = torch.randn(3, 6, 21)
    outputs = model(sequence, return_reconstruction=True)
    assert outputs["reconstruction"].shape == sequence.shape


def test_legacy_memory_rejects_nonfinite_runtime_updates():
    model = HierarchicalPatchTransformerCycleModel(
        input_dim=6,
        future_target_dim=4,
        d_model=16,
        nhead=4,
        num_layers=1,
        patch_size=5,
        latent_dim=8,
        num_memory_slots=4,
        dropout=0.0,
        memory_mode="legacy_ema",
        memory_update_rate=0.1,
    )

    class InfiniteWrite(torch.nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return torch.full(
                (values.size(0), 16),
                float("inf"),
                dtype=values.dtype,
                device=values.device,
            )

    model.write_net = InfiniteWrite()
    memory_before = model.memory_state.clone()
    outputs = model(torch.randn(3, 6, 21))

    assert torch.isfinite(outputs["future_pred"]).all()
    assert torch.isfinite(model.memory_state).all()
    assert torch.equal(model.memory_state, memory_before)
