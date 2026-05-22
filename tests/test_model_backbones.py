import torch

from src.models import build_cycle_model
from src.models.cycle_reasoning_model import CycleReasoningModel
from src.models.hierarchical_lstm_model import HierarchicalLSTMCycleModel


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


def test_build_cycle_model_supports_hierarchical_lstm():
    model = build_cycle_model(
        model_cfg={
            "encoder": "hierarchical_lstm",
            "lstm_hidden_dim": 8,
            "lstm_layers": 1,
            "patch_size": 5,
            "latent_dim": 16,
            "dropout": 0.0,
        },
        input_dim=6,
        future_target_dim=4,
    )

    assert isinstance(model, HierarchicalLSTMCycleModel)
    _assert_output_contract(model, latent_dim=16)

    sequence = torch.randn(3, 6, 21)
    outputs = model(sequence, return_reconstruction=True)
    assert outputs["reconstruction"].shape == sequence.shape
