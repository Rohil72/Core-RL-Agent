import pandas as pd

from src.backtest.market_memory_backtester import (
    PolicyConfig,
    compute_backtest_metrics,
    run_long_only_backtest,
)


def _signals():
    rows = []
    for date, aaa, bbb in [
        ("2024-01-01", 10.0, 20.0),
        ("2024-01-02", 11.0, 21.0),
        ("2024-01-03", 12.0, 22.0),
        ("2024-01-04", 9.0, 23.0),
    ]:
        rows.append(
            {
                "ticker": "AAA",
                "timestamp": pd.Timestamp(date, tz="UTC"),
                "open": aaa,
                "close": aaa,
                "opportunity_score": 0.20,
                "retrieval_expected_upside": 0.15,
                "retrieval_expected_downside": -0.03,
                "retrieval_confidence": 1.0,
            }
        )
        rows.append(
            {
                "ticker": "BBB",
                "timestamp": pd.Timestamp(date, tz="UTC"),
                "open": bbb,
                "close": bbb,
                "opportunity_score": 0.10,
                "retrieval_expected_upside": 0.12,
                "retrieval_expected_downside": -0.03,
                "retrieval_confidence": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_next_bar_execution_top_k_and_no_duplicate_ticker_entries():
    trades, _ = run_long_only_backtest(
        _signals(),
        PolicyConfig(top_k=1, min_hold_days=1, max_hold_days=2, slippage_bps=0, initial_capital=1000),
    )

    assert len(trades) >= 1
    assert trades.iloc[0]["ticker"] == "AAA"
    assert trades.iloc[0]["entry_date"] == pd.Timestamp("2024-01-02", tz="UTC")
    assert trades["ticker"].nunique() == 1


def test_slippage_reduces_trade_returns():
    no_slip, _ = run_long_only_backtest(
        _signals(),
        PolicyConfig(top_k=1, min_hold_days=1, max_hold_days=2, slippage_bps=0, initial_capital=1000),
    )
    slipped, _ = run_long_only_backtest(
        _signals(),
        PolicyConfig(top_k=1, min_hold_days=1, max_hold_days=2, slippage_bps=100, initial_capital=1000),
    )

    assert slipped.iloc[0]["net_return"] < no_slip.iloc[0]["net_return"]


def test_stop_loss_exit_and_metrics():
    trades, equity = run_long_only_backtest(
        _signals(),
        PolicyConfig(top_k=1, min_hold_days=5, max_hold_days=10, stop_loss=0.10, slippage_bps=0, initial_capital=1000),
    )
    metrics = compute_backtest_metrics(trades, equity, 1000)

    assert "stop_loss" in set(trades["exit_reason"])
    assert metrics["trade_count"] == len(trades)
    assert "max_drawdown" in metrics


def test_final_liquidation_uses_each_tickers_last_available_bar():
    frame = _signals()
    frame = frame[~((frame["ticker"] == "BBB") & (frame["timestamp"] == pd.Timestamp("2024-01-04", tz="UTC")))]
    trades, equity = run_long_only_backtest(
        frame,
        PolicyConfig(top_k=2, min_hold_days=5, max_hold_days=10, slippage_bps=0, initial_capital=1000),
    )

    assert {"AAA", "BBB"}.issubset(set(trades["ticker"]))
    assert trades[trades["ticker"] == "BBB"].iloc[-1]["exit_date"] == pd.Timestamp("2024-01-03", tz="UTC")
    assert equity.iloc[-1]["equity"] > 0


def test_entry_sizing_does_not_use_the_current_close_before_open_execution():
    rows = []
    for day in range(4):
        timestamp = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day)
        for ticker in ("AAA", "BBB"):
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "open": 100.0,
                    "close": 100.0,
                    "opportunity_score": 0.20 if ticker == "AAA" or day >= 1 else -1.0,
                    "retrieval_expected_upside": 0.15 if ticker == "AAA" or day >= 1 else -1.0,
                    "retrieval_expected_downside": -0.03,
                    "retrieval_confidence": 1.0,
                }
            )
    normal = pd.DataFrame(rows)
    distorted = normal.copy()
    mask = (distorted["ticker"] == "AAA") & (
        distorted["timestamp"] == pd.Timestamp("2024-01-03", tz="UTC")
    )
    distorted.loc[mask, "close"] = 10.0
    policy = PolicyConfig(
        top_k=2,
        min_hold_days=30,
        max_hold_days=60,
        stop_loss=0.95,
        slippage_bps=0,
        initial_capital=1000,
    )

    normal_trades, _ = run_long_only_backtest(normal, policy)
    distorted_trades, _ = run_long_only_backtest(distorted, policy)
    normal_bbb = normal_trades.loc[normal_trades["ticker"] == "BBB", "cost_basis"].iloc[-1]
    distorted_bbb = distorted_trades.loc[distorted_trades["ticker"] == "BBB", "cost_basis"].iloc[-1]

    assert normal_bbb == distorted_bbb == 500.0


def test_entry_and_exit_scores_can_be_decoupled_for_rally_start_policies():
    rows = []
    for day, rally_score in enumerate((1.0, 0.0, 0.0, 0.0)):
        rows.append(
            {
                "ticker": "AAA",
                "timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day),
                "open": 100.0,
                "close": 100.0,
                "opportunity_score": 0.20,
                "rally_entry_score": rally_score,
                "retrieval_expected_upside": 0.15,
                "retrieval_expected_downside": -0.03,
                "retrieval_confidence": 1.0,
            }
        )
    signals = pd.DataFrame(rows)
    policy = PolicyConfig(
        top_k=1,
        min_hold_days=1,
        max_hold_days=10,
        stop_loss=0.95,
        slippage_bps=0,
        initial_capital=1000,
    )

    coupled, _ = run_long_only_backtest(signals, policy, score_col="rally_entry_score")
    decoupled, _ = run_long_only_backtest(
        signals,
        policy,
        score_col="rally_entry_score",
        exit_score_col="opportunity_score",
    )

    assert coupled.iloc[0]["exit_reason"] == "score_decay"
    assert decoupled.iloc[0]["exit_reason"] == "end_of_test"
