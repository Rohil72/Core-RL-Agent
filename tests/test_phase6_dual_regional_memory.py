from pathlib import Path

import torch
import yaml
from torch import nn

from src.decision.adapter import DecisionAdapter
from src.decision.alignment import (
    AlignedDecisionModel,
    AlignmentConfig,
    configure_alignment_parameters,
)
from scripts.run_phase5_transformer_alignment import build_transfer_encoder_payload


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(4, 4)
        self.memory_slots = nn.Parameter(torch.ones(2, 4))
        self.memory_attn = nn.Linear(4, 4)
        self.memory_query_proj = nn.Linear(4, 4)
        self.context = nn.Linear(4, 4)
        self.future_head = nn.Linear(4, 4)


def test_memory_only_alignment_freezes_global_backbone():
    model = AlignedDecisionModel(_Encoder(), DecisionAdapter())

    encoder_parameters, adapter_parameters = configure_alignment_parameters(model, "memory_only")
    trainable = {name for name, parameter in model.encoder.named_parameters() if parameter.requires_grad}

    assert trainable == {
        "memory_slots",
        "memory_attn.weight",
        "memory_attn.bias",
        "memory_query_proj.weight",
        "memory_query_proj.bias",
    }
    assert len(encoder_parameters) == len(trainable)
    assert adapter_parameters


def test_dual_regional_contract_keeps_one_global_backbone_and_two_regional_memories():
    config = yaml.safe_load(Path("configs/phase6_dual_regional_memory.yaml").read_text(encoding="utf-8"))
    topology = config["topologies"][0]

    assert topology == {
        "id": "global_dual_regional",
        "transformer_backbone": "global_frozen",
        "internal_memory": "regional",
        "external_memory": "regional",
    }
    assert config["alignment"]["encoder_parameter_scope"] == "memory_only"
    assert config["alignment"]["epochs"] == 2
    assert len(config["experiment"]["seeds"]) == 3


def test_alignment_scope_rejects_accidental_full_encoder_training():
    try:
        AlignmentConfig(encoder_parameter_scope="full_encoder")
    except ValueError as exc:
        assert "encoder_parameter_scope" in str(exc)
    else:
        raise AssertionError("An unsupported alignment scope must fail closed.")


def test_transfer_checkpoint_reloads_the_adapted_static_memory_contract():
    encoder = _Encoder()
    payload = {
        "model_config": {"encoder": "patch_transformer", "memory_mode": "legacy_ema"},
        "model_state": {"stale": torch.tensor(0.0)},
    }

    transfer = build_transfer_encoder_payload(payload, encoder)

    assert transfer["model_config"]["memory_mode"] == "static_parameter"
    assert payload["model_config"]["memory_mode"] == "legacy_ema"
    assert torch.equal(transfer["model_state"]["memory_slots"], encoder.memory_slots)
