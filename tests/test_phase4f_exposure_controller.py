import json
from pathlib import Path

import yaml

from scripts.run_phase4f_exposure_controller import run_confirmation
from src.backtest.exposure_controller import ExposureControllerConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config():
    return yaml.safe_load((PROJECT_ROOT / "configs" / "phase4f_exposure_controller.yaml").read_text())


def test_phase4f_candidates_are_unique_and_resolve():
    config = _config()
    identifiers = [item["id"] for item in config["controllers"]]

    assert identifiers == [
        "F0_a2_static",
        "F1_volatility",
        "F2_drawdown",
        "F3_evidence",
        "F4_combined",
    ]
    assert len(identifiers) == len(set(identifiers))
    for item in config["controllers"]:
        assert isinstance(ExposureControllerConfig(**item["config"]), ExposureControllerConfig)


def test_phase4f_confirmation_stays_sealed_after_failed_validation(tmp_path):
    selection = {
        "confirmation_unlocked": False,
        "smoke_only": False,
        "failure_reason": "cross-market gate failed",
    }

    run_confirmation(_config(), tmp_path, selection, resume=False)

    status = json.loads((tmp_path / "confirmation_status.json").read_text())
    assert status == {"executed": False, "reason": "cross-market gate failed"}
