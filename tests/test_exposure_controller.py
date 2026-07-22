import numpy as np
import pandas as pd
from types import SimpleNamespace

from src.backtest.exposure_controller import (
    CausalExposureController,
    ExposureControllerConfig,
    ExposureDecision,
)
from src.backtest.market_memory_backtester import PolicyConfig, run_long_only_backtest


def _evidence(confidence=0.60, agreement=0.50, ess=20.0, positive=True):
    return pd.DataFrame(
        {
            "retrieval_confidence": [confidence, confidence],
            "retrieval_agreement_score": [agreement, agreement],
            "retrieval_effective_sample_size": [ess, ess],
            "retrieval_alpha_ci_low": [0.01 if positive else -0.01, 0.02 if positive else -0.02],
            "retrieval_ood_pass": [True, True],
        }
    )


def test_disabled_controller_preserves_full_exposure():
    controller = CausalExposureController(ExposureControllerConfig(enabled=False))

    decision = controller.decide(_evidence(), pd.DataFrame())

    assert decision.target_exposure == 1.0
    assert decision.reason == "disabled"


def test_volatility_target_uses_only_prior_equity_and_never_leverages():
    equity = pd.DataFrame({"equity": 100 * np.cumprod(1 + np.array([0.0, 0.04, -0.04] * 8))})
    controller = CausalExposureController(
        ExposureControllerConfig(
            target_annualized_volatility=0.15,
            volatility_lookback=20,
            volatility_min_observations=10,
            drawdown_soft_limit=None,
            drawdown_hard_limit=None,
            evidence_enabled=False,
        )
    )

    decision = controller.decide(_evidence(), equity)

    assert decision.realized_annualized_volatility is not None
    assert 0.0 < decision.target_exposure < 1.0
    assert decision.target_exposure == decision.volatility_scalar


def test_drawdown_and_weak_memory_reduce_exposure_smoothly():
    equity = pd.DataFrame({"equity": [100.0, 110.0, 105.0, 90.0]})
    controller = CausalExposureController(
        ExposureControllerConfig(
            target_annualized_volatility=None,
            drawdown_soft_limit=0.10,
            drawdown_hard_limit=0.25,
            evidence_enabled=True,
        )
    )

    decision = controller.decide(_evidence(0.15, 0.10, 5.0, positive=False), equity)

    assert decision.current_drawdown < -0.10
    assert 0.25 <= decision.drawdown_scalar < 1.0
    assert 0.35 <= decision.evidence_scalar < 1.0
    assert decision.target_exposure == min(decision.drawdown_scalar, decision.evidence_scalar)


def test_backtester_records_causal_exposure_decisions():
    rows = []
    for day in range(16):
        timestamp = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day)
        price = 100.0 * (1.04 if day % 2 else 0.96)
        rows.append(
            {
                "timestamp": timestamp,
                "ticker": "AAA",
                "open": price,
                "close": price,
                "opportunity_score": 0.10,
                "retrieval_expected_upside": 0.15,
                "retrieval_expected_downside": -0.03,
                "retrieval_confidence": 0.60,
                "retrieval_agreement_score": 0.50,
                "retrieval_effective_sample_size": 20.0,
                "retrieval_alpha_ci_low": 0.01,
                "retrieval_ood_pass": True,
            }
        )
    log = []
    controller = CausalExposureController(
        ExposureControllerConfig(
            target_annualized_volatility=0.10,
            volatility_lookback=5,
            volatility_min_observations=3,
            drawdown_soft_limit=None,
            drawdown_hard_limit=None,
            evidence_enabled=False,
            rebalance_threshold=0.0,
        )
    )

    trades, equity = run_long_only_backtest(
        pd.DataFrame(rows),
        PolicyConfig(top_k=1, min_hold_days=30, max_hold_days=60, slippage_bps=0),
        exposure_controller=controller,
        exposure_log=log,
    )

    assert log
    assert min(item["target_exposure"] for item in log) < 1.0
    assert equity["target_exposure"].between(0.0, 1.0).all()
    assert "trade_id" in trades


def test_backtester_can_restore_exposure_without_waiting_for_a_new_entry():
    rows = []
    for day in range(4):
        rows.append(
            {
                "timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day),
                "ticker": "AAA",
                "open": 100.0,
                "close": 100.0,
                "opportunity_score": 0.10,
                "retrieval_expected_upside": 0.15,
                "retrieval_expected_downside": -0.03,
                "retrieval_confidence": 1.0,
            }
        )

    class ScheduledController:
        config = SimpleNamespace(rebalance_threshold=0.0)

        def __init__(self):
            self.index = 0

        def decide(self, signal_rows, prior_equity):
            del signal_rows, prior_equity
            target = [0.5, 1.0, 1.0][self.index]
            self.index += 1
            return ExposureDecision(target, 1.0, 1.0, 1.0, None, 0.0, None, None, None, None, "scheduled")

    trades, _ = run_long_only_backtest(
        pd.DataFrame(rows),
        PolicyConfig(top_k=1, min_hold_days=30, max_hold_days=60, slippage_bps=0),
        exposure_controller=ScheduledController(),
    )

    assert len(trades) == 1
    assert trades.iloc[0]["cost_basis"] > 99000.0


def test_top_candidate_consensus_budget_is_invariant_to_score_scale():
    frame = pd.DataFrame(
        {
            "consensus_entry_rank": [0.95, 0.80, 0.20, 0.10],
            "retrieval_confidence": [0.65, 0.62, 0.20, 0.15],
            "retrieval_agreement_score": [0.60, 0.55, 0.10, 0.10],
            "retrieval_effective_sample_size": [24.8, 24.7, 10.0, 8.0],
            "retrieval_alpha_ci_low": [0.03, 0.02, -0.04, -0.05],
            "retrieval_ood_pass": [True, True, True, True],
            "seed_vote_fraction": [1.0, 2 / 3, 0.0, 0.0],
            "seed_rank_std": [0.02, 0.04, 0.20, 0.25],
            "seed_score_positive_fraction": [1.0, 2 / 3, 0.0, 0.0],
        }
    )
    controller = CausalExposureController(
        ExposureControllerConfig(
            target_annualized_volatility=None,
            drawdown_soft_limit=None,
            drawdown_hard_limit=None,
            evidence_enabled=True,
            evidence_top_fraction=0.50,
            evidence_score_column="consensus_entry_rank",
            seed_consensus_enabled=True,
            evidence_floor_scalar=0.25,
        )
    )

    original = controller.decide(frame, pd.DataFrame())
    rescaled = frame.copy()
    rescaled["consensus_entry_rank"] = 1000.0 + 50.0 * rescaled["consensus_entry_rank"]
    transformed = controller.decide(rescaled, pd.DataFrame())

    assert original.target_exposure == transformed.target_exposure
    assert original.evidence_candidate_count == 2
    assert original.mean_seed_vote_fraction > 0.80


def test_seed_disagreement_derisks_but_preserves_recovery_exposure():
    strong = _evidence()
    strong["consensus_entry_rank"] = [1.0, 0.9]
    strong["seed_vote_fraction"] = [1.0, 1.0]
    strong["seed_rank_std"] = [0.01, 0.02]
    strong["seed_score_positive_fraction"] = [1.0, 1.0]
    weak = strong.copy()
    weak["seed_vote_fraction"] = 0.0
    weak["seed_rank_std"] = 0.25
    weak["seed_score_positive_fraction"] = 0.0
    controller = CausalExposureController(
        ExposureControllerConfig(
            target_annualized_volatility=None,
            drawdown_soft_limit=None,
            drawdown_hard_limit=None,
            evidence_enabled=True,
            evidence_score_column="consensus_entry_rank",
            seed_consensus_enabled=True,
            evidence_floor_scalar=0.35,
            minimum_exposure=0.25,
        )
    )

    strong_decision = controller.decide(strong, pd.DataFrame())
    weak_decision = controller.decide(weak, pd.DataFrame())

    assert 0.25 <= weak_decision.target_exposure < strong_decision.target_exposure
    assert weak_decision.reason == "evidence"
