from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.train_phase5_offline_policy import _d3rlpy_train


class _FakeLearner:
    def __init__(self) -> None:
        self.fit_dataset = None
        self.saved_output = None

    def fit(self, dataset, **kwargs) -> None:
        self.fit_dataset = dataset

    def save(self, output: str) -> None:
        self.saved_output = output


@pytest.mark.parametrize("algorithm", ["cql", "iql", "td3bc"])
def test_d3rlpy_training_declares_continuous_exposure_actions(
    monkeypatch, algorithm: str
) -> None:
    captured = {}
    learner = _FakeLearner()

    def dataset_factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    class _FakeConfig:
        def __init__(self, **kwargs) -> None:
            captured["config"] = kwargs

        def create(self, **kwargs):
            captured["device"] = kwargs["device"]
            return learner

    continuous = object()
    fake_d3rlpy = SimpleNamespace(
        dataset=SimpleNamespace(MDPDataset=dataset_factory),
        constants=SimpleNamespace(ActionSpace=SimpleNamespace(CONTINUOUS=continuous)),
        algos=SimpleNamespace(
            CQLConfig=_FakeConfig,
            IQLConfig=_FakeConfig,
            TD3PlusBCConfig=_FakeConfig,
        ),
    )
    monkeypatch.setitem(sys.modules, "d3rlpy", fake_d3rlpy)
    data = {
        "observations": np.zeros((5, 3), dtype=np.float32),
        "actions": np.asarray([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=np.float32),
        "rewards": np.zeros(5, dtype=np.float32),
        "terminals": np.asarray([0, 0, 0, 0, 1], dtype=np.float32),
    }
    output = Path(f"{algorithm}.d3")

    _d3rlpy_train(algorithm, data, output, steps=10, device="cpu")

    assert captured["action_space"] is continuous
    assert captured["action_size"] == 1
    assert captured["actions"].shape == (5, 1)
    assert learner.fit_dataset is not None
    assert learner.saved_output == str(output)
