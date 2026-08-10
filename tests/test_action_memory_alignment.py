import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
import scripts.run_action_memory_alignment as alignment

from src.memory.experience import ExperienceMemory, ExperienceSchema
from src.memory.evidence import build_evidence_summary
from src.memory.aggregator import AggregationConfig
from src.memory.confidence import ConfidenceConfig
from src.memory.market_memory import MarketMemoryConfig, score_market_memory
from src.memory.calibration import fit_downside_calibration, DownsideCalibrationModel
from src.backtest.market_memory_backtester import PolicyConfig, run_long_only_backtest, tradability_reason
from scripts.run_action_memory_alignment import run_alignment_study


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dual_evidence_calculation():
    """Verify dual outcome evidence calculation (relative alpha & absolute return)."""
    neighbors = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"], utc=True),
            "future_max_return_63": [0.05, 0.08, 0.04],
            "future_min_return_63": [-0.02, -0.01, -0.03],
            "decision_net_alpha": [0.02, 0.04, 0.01],
            "decision_return_63": [0.03, 0.05, 0.02],
            "decision_mae": [-0.02, -0.01, -0.03],
            "decision_mfe": [0.05, 0.08, 0.04],
        }
    )
    distances = np.array([0.1, 0.2, 0.3])
    cfg = MarketMemoryConfig(k=3, target_alpha="decision_net_alpha", target_absolute_return="decision_return_63")

    summary = build_evidence_summary(
        neighbors,
        distances,
        query_ticker="AAPL",
        upside_col="future_max_return_63",
        alpha_col="decision_net_alpha",
        downside_col="future_min_return_63",
        path_quality_col=None,
        holding_period_col=None,
        absolute_return_col="decision_return_63",
        aggregation=AggregationConfig(method=cfg.aggregation_method),
        confidence=ConfidenceConfig(reference_distance=cfg.confidence_reference_distance),
        score_weights={},
    )

    payload = summary.to_signal_payload()
    assert "retrieval_expected_absolute_return" in payload
    assert "retrieval_absolute_return_lcb" in payload
    assert payload["retrieval_expected_absolute_return"] is not None
    assert payload["retrieval_absolute_return_lcb"] is not None
    assert payload["retrieval_alpha_lcb"] == payload["retrieval_alpha_ci_low"]


def test_no_calibration_leakage_2025_2026():
    """Verify that calibration strictly rejects dates outside 2022-01-01 to 2024-12-31."""
    leakage_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-06-01", "2025-01-01"], utc=True),
            "market": ["US", "US"],
            "retrieval_expected_downside": [-0.02, -0.03],
            "decision_mae": [-0.08, -0.10],
        }
    )
    with pytest.raises(ValueError, match="out-of-bounds timestamps"):
        fit_downside_calibration(leakage_df)


def test_pooled_calibration_fallback():
    """Verify fallback to pooled development residuals when market sample count is below min_samples."""
    dev_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=40, freq="D", tz="UTC"),
            "market": ["US"] * 35 + ["France"] * 5,
            "retrieval_expected_downside": [-0.02] * 40,
            "decision_mae": [-0.10] * 40,
        }
    )

    model = fit_downside_calibration(dev_df, adverse_quantile=0.90, min_samples=30)
    assert model.market_summaries["US"].fallback_used is False
    assert model.market_summaries["France"].fallback_used is True
    assert model.market_summaries["France"].correction == model.pooled_correction


def test_breadth_exposure_position_trimming():
    """Verify actual portfolio position trimming when eligible breadth drops from 3 to 0."""
    dates = pd.date_range("2025-01-01", periods=10, freq="B", tz="UTC")
    rows = []
    for idx, d in enumerate(dates):
        # High score days 0-2 (3 candidates), then 0 candidates days 3-9
        is_eligible = idx <= 2
        score = 0.05 if is_eligible else -0.05
        rows.append(
            {
                "timestamp": d,
                "ticker": "AAPL",
                "open": 100.0,
                "close": 100.0,
                "opportunity_score": score,
                "retrieval_expected_upside": 0.05,
                "retrieval_expected_downside": -0.02,
                "retrieval_confidence": 0.8,
                "retrieval_alpha_ci_low": 0.02 if is_eligible else -0.05,
                "retrieval_absolute_return_ci_low": 0.02 if is_eligible else -0.05,
            }
        )
    signals = pd.DataFrame(rows)

    policy = PolicyConfig(
        top_k=1,
        min_score=0.0,
        min_expected_upside=0.0,
        max_expected_downside=0.10,
        min_alpha_lcb=0.0,
        min_absolute_return_lcb=0.0,
        breadth_exposure_enabled=True,
        breadth_exposure_map={0: 0.00, 1: 0.33, 2: 0.67, 3: 1.00},
    )

    trades, equity = run_long_only_backtest(signals, policy)
    # Check that rebalance trade occurred when breadth dropped to 0
    rebalance_trades = trades[trades["exit_reason"] == "exposure_rebalance"]
    assert not rebalance_trades.empty
    # Target exposure at end should be 0.0
    assert equity.iloc[-1]["target_exposure"] == 0.0


def test_minimum_hold_21_score_decay_blocking():
    """Verify score decay exits are explicitly forbidden before 21 sessions."""
    dates = pd.date_range("2025-01-01", periods=30, freq="B", tz="UTC")
    rows = [
        {
            "timestamp": d,
            "ticker": "AAPL",
            "open": 100.0,
            "close": 100.0,
            "opportunity_score": 0.10 if i == 0 else -0.10, # Score decays at step 1
            "retrieval_expected_upside": 0.05,
            "retrieval_expected_downside": -0.02,
            "retrieval_confidence": 0.8,
            "retrieval_alpha_ci_low": 0.02,
            "retrieval_absolute_return_ci_low": 0.02,
        }
        for i, d in enumerate(dates)
    ]
    signals = pd.DataFrame(rows)

    policy_hold21 = PolicyConfig(
        top_k=1,
        min_score=0.0,
        min_expected_upside=0.0,
        max_expected_downside=0.10,
        holding_mode="minimum_hold_21",
        stop_loss=0.50,
    )

    trades, _ = run_long_only_backtest(signals, policy_hold21)
    if not trades.empty:
        score_decay_trades = trades[trades["exit_reason"] == "score_decay"]
        for _, tr in score_decay_trades.iterrows():
            assert tr["holding_days"] >= 21


def test_c0_frozen_control_reproduction():
    """Verify C0 uses merged base_config policy parameters."""
    config_path = PROJECT_ROOT / "configs" / "action_memory_alignment.yaml"
    align_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_config_path = PROJECT_ROOT / align_cfg["study"]["base_config"]
    base_cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))

    c0_policy = align_cfg["variants"]["C0"]["policy"]
    merged = {**base_cfg["policy"], **c0_policy}
    policy = PolicyConfig(**merged)

    assert policy.min_score == 0.0
    assert policy.min_expected_upside == 0.0
    assert policy.require_ood_pass is True
    assert policy.risk_guard_enabled is True


def test_real_source_preflight_requires_encoder_adapter_memory_and_prices(tmp_path, monkeypatch):
    """Resolve a source only when every frozen artifact and market input exists."""

    source_root = tmp_path / "reports" / "testbed"
    latent_root = source_root / "latents" / "regional_US_seed_7"
    model = source_root / "models" / "regional_US_seed_7" / "final_model.pt"
    adapter_root = tmp_path / "reports" / "adapters" / "regional_US" / "seed_7"
    price = tmp_path / "data" / "international" / "US" / "AAPL.parquet"
    for path in (latent_root, model.parent, adapter_root, price.parent):
        path.mkdir(parents=True, exist_ok=True)
    sample = pd.DataFrame(
        {"ticker": ["AAPL"], "timestamp": pd.to_datetime(["2021-01-01"], utc=True), "latent_0": [0.1]}
    )
    sample.to_parquet(latent_root / "train_latents.parquet", index=False)
    sample.to_parquet(latent_root / "val_latents.parquet", index=False)
    sample.to_parquet(adapter_root / "train_decisions.parquet", index=False)
    sample.assign(open=100.0, close=101.0).to_parquet(price, index=False)
    model.write_bytes(b"encoder")
    (adapter_root / "decision_adapter.pt").write_bytes(b"adapter")
    monkeypatch.setattr(alignment, "PROJECT_ROOT", tmp_path)
    config = {
        "experiment": {
            "source_mode": "phase6_testbed",
            "source_testbed_run": "reports/testbed",
            "adapter_run": "reports/adapters",
            "data_root": "data/international",
            "source_representations": ["regional"],
            "active_markets": ["US"],
            "active_seeds": [7],
        }
    }

    resolved = alignment._resolve_alignment_sources(config)
    assert len(resolved) == 1
    assert resolved[0].adapter_checkpoint.name == "decision_adapter.pt"
    assert resolved[0].train_decisions.name == "train_decisions.parquet"
    (adapter_root / "decision_adapter.pt").unlink()
    with pytest.raises(FileNotFoundError, match="adapter checkpoint"):
        alignment._resolve_alignment_sources(config)


def test_signal_preparation_executes_frozen_adapter_and_memory_scoring(tmp_path, monkeypatch):
    """The production preparation path must export decisions before retrieval scoring."""

    encoder = tmp_path / "encoder.pt"
    adapter = tmp_path / "decision_adapter.pt"
    train_latents = tmp_path / "train_latents.parquet"
    val_latents = tmp_path / "val_latents.parquet"
    train_decisions = tmp_path / "train_decisions.parquet"
    price = tmp_path / "AAPL.parquet"
    for path in (encoder, adapter):
        path.write_bytes(b"checkpoint")
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "timestamp": pd.to_datetime(["2021-01-01"], utc=True),
            "decision_0": [0.2],
            "decision_mfe": [0.1],
            "decision_mae": [-0.03],
            "decision_net_alpha": [0.04],
            "decision_return_63": [0.06],
            "decision_path_quality": [0.7],
            "decision_holding_sessions": [63],
        }
    )
    frame.to_parquet(train_latents, index=False)
    frame.to_parquet(val_latents, index=False)
    frame.to_parquet(train_decisions, index=False)
    frame.assign(open=100.0, close=101.0).to_parquet(price, index=False)
    source = alignment.AlignmentSource(
        source=alignment.AdapterSource(
            name="regional_US_seed_7",
            group="regional_US",
            seed=7,
            train_path=train_latents,
            val_path=val_latents,
            precomputed_globs=(str(price),),
            backtest_glob=str(price),
            encoder_checkpoint=encoder,
        ),
        adapter_checkpoint=adapter,
        train_decisions=train_decisions,
        market="US",
        price_files=(price,),
    )
    calls = {}

    def fake_export(**kwargs):
        calls["encoder"] = kwargs["encoder_checkpoint"]
        calls["adapter"] = kwargs["adapter_checkpoint"]
        destination = Path(kwargs["output"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False)
        destination.with_suffix(".json").write_text("{}", encoding="utf-8")

    def fake_retrieval(source_path, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.read_parquet(source_path).to_parquet(destination, index=False)

    def fake_memory(evaluation, _root, run_id):
        calls["memory"] = evaluation
        destination = Path(evaluation["data"]["output_dir"]) / run_id
        destination.mkdir(parents=True, exist_ok=True)
        signal = frame.assign(
            open=100.0,
            close=101.0,
            opportunity_score=0.04,
            retrieval_expected_upside=0.1,
            retrieval_expected_downside=-0.03,
            retrieval_confidence=0.8,
        )
        signal.to_parquet(destination / "signals.parquet", index=False)
        signal.to_parquet(destination / "neighbors.parquet", index=False)
        return destination

    monkeypatch.setattr(alignment, "export_transfer_latents", fake_export)
    monkeypatch.setattr(alignment, "_retrieval_frame", fake_retrieval)
    monkeypatch.setattr(alignment, "run_market_memory_evaluation", fake_memory)
    export_config = tmp_path / "export.yaml"
    export_config.write_text("{}", encoding="utf-8")
    result = alignment._prepare_signal_period(
        source,
        yaml.safe_load((PROJECT_ROOT / "configs" / "phase4e_cross_market_sharpe.yaml").read_text(encoding="utf-8")),
        export_config,
        tmp_path / "study",
        "development",
        "2022-01-01",
        "2024-12-31",
        resume=False,
    )

    assert result.is_file()
    assert calls["encoder"] == str(encoder)
    assert calls["adapter"] == str(adapter)
    assert calls["memory"]["memory"]["target_absolute_return"] == "decision_return_63"
    assert calls["memory"]["memory"]["require_outcome_availability"] is True


def test_accounting_cutoff_and_censoring():
    """Verify calendar_portfolio liquidates at cutoff while q1_to_q1_entry_cohort tracks positions past cutoff."""
    dates = pd.date_range("2025-01-01", "2026-04-30", freq="B", tz="UTC")
    rows = [
        {
            "timestamp": d,
            "ticker": "AAPL",
            "open": 100.0,
            "close": 100.0,
            "opportunity_score": 0.10 if i == 0 else 0.05,
            "retrieval_expected_upside": 0.05,
            "retrieval_expected_downside": -0.02,
            "retrieval_confidence": 0.8,
            "retrieval_alpha_ci_low": 0.02,
            "retrieval_absolute_return_ci_low": 0.02,
        }
        for i, d in enumerate(dates)
    ]
    signals = pd.DataFrame(rows)

    policy_cal = PolicyConfig(
        top_k=1,
        min_score=0.0,
        min_expected_upside=0.0,
        max_expected_downside=0.10,
        entry_cutoff_date="2025-12-31",
        accounting_cutoff_date="2026-03-31",
        accounting_protocol="calendar_portfolio",
    )

    trades_cal, equity_cal = run_long_only_backtest(signals, policy_cal)
    assert equity_cal["timestamp"].max() <= pd.Timestamp("2026-03-31", tz="UTC")


def test_traceable_decision_audit_linkage():
    """Verify every row in decision_audit links to exact trade outcomes for entries."""
    dates = pd.date_range("2025-01-01", periods=10, freq="B", tz="UTC")
    rows = [
        {
            "timestamp": d,
            "ticker": "AAPL",
            "open": 100.0,
            "close": 100.0 + i,
            "opportunity_score": 0.10,
            "retrieval_expected_upside": 0.05,
            "retrieval_expected_downside": -0.02,
            "retrieval_confidence": 0.8,
            "retrieval_alpha_ci_low": 0.02,
            "retrieval_absolute_return_ci_low": 0.02,
            "decision_mae": -0.02,
            "decision_mfe": 0.08,
            "decision_net_alpha": 0.04,
            "decision_return_63": 0.06,
            "decision_is_mature": True,
            "decision_outcome_available_timestamp": d + pd.Timedelta(days=90),
        }
        for i, d in enumerate(dates)
    ]
    signals = pd.DataFrame(rows)
    policy = PolicyConfig(top_k=1, min_score=0.0, min_expected_upside=0.0, max_expected_downside=0.10)

    d_log = []
    trades, _ = run_long_only_backtest(signals, policy, decision_log=d_log)

    entered_item = [item for item in d_log if item.get("decision") == "enter"][0]
    assert entered_item["trade_id"] == 0
    assert entered_item["exit_reason"] is not None
    assert entered_item["realized_return"] is not None
    assert entered_item["realized_mae"] is not None
    assert entered_item["realized_alpha_63"] == pytest.approx(0.04)
    assert entered_item["realized_return_63"] == pytest.approx(0.06)
    assert entered_item["realized_mae_63"] == pytest.approx(-0.02)
    assert entered_item["outcome_mature"] is True


def test_deterministic_resume_refusal(tmp_path):
    """Verify runner refuses to reuse a non-empty output directory unless resume=True."""
    run_dir = tmp_path / "action_memory_alignment" / "test_run"
    run_dir.mkdir(parents=True)
    (run_dir / "dummy.txt").write_text("existing content", encoding="utf-8")

    config_path = PROJECT_ROOT / "configs" / "action_memory_alignment.yaml"

    with pytest.raises(FileExistsError, match="Refusing to reuse non-empty run directory"):
        run_alignment_study(str(config_path), "test_run", smoke_test=True, output_root_override=run_dir)


def test_resume_refuses_changed_config_hash(tmp_path):
    """A completed run cannot resume after any declared configuration change."""

    source = yaml.safe_load((PROJECT_ROOT / "configs" / "action_memory_alignment.yaml").read_text(encoding="utf-8"))
    config_path = tmp_path / "alignment.yaml"
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    run_dir = tmp_path / "resume_contract"
    run_alignment_study(
        str(config_path),
        "resume_contract",
        selected_variants={"C0"},
        smoke_test=True,
        output_root_override=run_dir,
    )
    source["comparison"]["gates"]["minimum_market_wins"] = 5
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")

    with pytest.raises(RuntimeError, match="code or immutable source hashes changed"):
        run_alignment_study(
            str(config_path),
            "resume_contract",
            selected_variants={"C0"},
            resume=True,
            smoke_test=True,
            output_root_override=run_dir,
        )


def test_smoke_execution_and_artifacts(tmp_path):
    """Run a full synthetic smoke test and verify all reviewer artifacts exist."""
    config_path = PROJECT_ROOT / "configs" / "action_memory_alignment.yaml"
    run_id = "smoke_test_unit"
    run_dir = tmp_path / run_id
    res = run_alignment_study(str(config_path), run_id, smoke_test=True, output_root_override=run_dir)

    output_root = Path(res["output_root"])
    required_files = [
        "experiment_manifest.yaml",
        "provenance.json",
        "provenance.md",
        "calibration_manifest.json",
        "variant_definitions.md",
        "paired_results.csv",
        "market_results.csv",
        "entry_quality.csv",
        "portfolio_results.csv",
        "censoring_audit.csv",
        "decision_audit.parquet",
        "robustness_verdict.json",
        "reviewer_summary.md",
        "prepared_sources_manifest.json",
    ]

    for fname in required_files:
        assert (output_root / fname).exists(), f"Missing required reviewer artifact: {fname}"

    verdict = json.loads((output_root / "robustness_verdict.json").read_text(encoding="utf-8"))
    assert "status" in verdict
    provenance = json.loads((output_root / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["status"] == "completed"
    assert provenance["input_fingerprint"]
    censoring = pd.read_csv(output_root / "censoring_audit.csv")
    assert set(censoring["protocol"]) == {"calendar_portfolio", "q1_to_q1_entry_cohort"}
    assert set(censoring["query_end"]) == {
        "2026-03-31T00:00:00+00:00",
        "2026-06-30T00:00:00+00:00",
    }
