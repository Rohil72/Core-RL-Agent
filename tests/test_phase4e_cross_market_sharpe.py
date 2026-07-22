import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.run_phase4e_cross_market_sharpe import (
    _policy_for_candidate,
    _sharpe,
    run_confirmation,
)
from src.backtest.market_memory_backtester import PolicyConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config():
    return yaml.safe_load((PROJECT_ROOT / "configs" / "phase4e_cross_market_sharpe.yaml").read_text())


def test_every_predeclared_policy_resolves_to_a_valid_policy_config():
    config = _config()
    identifiers = [candidate["id"] for candidate in config["policy_candidates"]]

    assert len(identifiers) == len(set(identifiers))
    assert {variant["id"] for variant in config["variants"]} == {
        "A0_legacy_identity",
        "A1_universe_alpha_identity",
        "A2_blended_alpha_identity",
    }
    for candidate in config["policy_candidates"]:
        resolved = _policy_for_candidate(config, candidate)
        assert isinstance(PolicyConfig(**resolved), PolicyConfig)
    assert _policy_for_candidate(config, config["policy_candidates"][0])["min_alpha_lcb"] is None


def test_annualized_sharpe_uses_daily_portfolio_returns():
    daily = pd.Series([0.01, -0.005, 0.012, -0.002, 0.008])
    expected = daily.mean() / daily.std(ddof=1) * np.sqrt(252.0)

    assert np.isclose(_sharpe(daily), expected)


def test_confirmation_remains_locked_when_promotion_fails(tmp_path):
    selection = {
        "confirmation_unlocked": False,
        "smoke_only": False,
        "failure_reason": "cross-market gate failed",
    }

    run_confirmation({}, tmp_path, [], selection, resume=False)

    status = json.loads((tmp_path / "confirmation_status.json").read_text())
    assert status == {"executed": False, "reason": "cross-market gate failed"}
