from pathlib import Path

import yaml

from src.backtest.exposure_controller import ExposureControllerConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase5_consensus_exposure_sweep_is_small_and_scale_free():
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "phase5_consensus_exposure.yaml").read_text(encoding="utf-8")
    )
    identifiers = [item["id"] for item in config["controllers"]]

    assert identifiers == [
        "E0_c0_static",
        "E1_volatility_15",
        "E2_volatility_drawdown",
        "E3_consensus_risk_budget",
    ]
    assert len(identifiers) == len(set(identifiers))
    resolved = {item["id"]: ExposureControllerConfig(**item["config"]) for item in config["controllers"]}
    assert resolved["E1_volatility_15"].maximum_exposure == 1.0
    assert resolved["E3_consensus_risk_budget"].evidence_score_column == "consensus_entry_rank"
    assert resolved["E3_consensus_risk_budget"].evidence_top_fraction == 0.25
    assert resolved["E3_consensus_risk_budget"].minimum_exposure > 0.0


def test_phase5_consensus_exposure_requires_cross_fold_robustness():
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "phase5_consensus_exposure.yaml").read_text(encoding="utf-8")
    )
    gates = config["selection"]

    assert gates["minimum_pooled_sharpe"] >= 2.0
    assert gates["minimum_cross_fold_oof_sharpe"] >= 2.0
    assert gates["minimum_positive_fold_fraction"] >= 0.8
    assert gates["maximum_worst_drawdown"] <= 0.20
