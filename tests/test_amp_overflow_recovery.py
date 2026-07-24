from __future__ import annotations

import torch

from src.trainers.train_cycle_model import _recover_amp_overflow


class _RecoveringScaler:
    def __init__(self) -> None:
        self.scale = 1024.0
        self.step_calls = 0
        self.update_calls = 0

    def get_scale(self) -> float:
        return self.scale

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.step_calls += 1

    def update(self) -> None:
        self.update_calls += 1
        self.scale /= 2.0


def test_amp_overflow_recovery_skips_update_and_reduces_scale() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter.grad = torch.tensor([float("inf")])
    scaler = _RecoveringScaler()

    previous_scale, updated_scale = _recover_amp_overflow(scaler, optimizer)

    assert previous_scale == 1024.0
    assert updated_scale == 512.0
    assert scaler.step_calls == 1
    assert scaler.update_calls == 1
    assert parameter.grad is None
