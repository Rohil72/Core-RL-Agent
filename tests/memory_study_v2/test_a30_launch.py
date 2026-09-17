"""Tests for A30 production deployment launcher (Finding 5 / C6).

Verifies:
1. Authorized launch calls driver_fn exactly once and records execution.
2. In-memory configuration authorization preserves repository config guard (production_authorized: false).
3. Missing required storage/manifests immediately fails closed without calling driver.
4. Driver failure records PRODUCTION_FAILED receipt and propagates error.
5. Unauthorized invocation without --authorize-production does not execute driver.
6. Check-only mode completes preflight checks without calling driver.
7. Corrupted repository config with production_authorized: true immediately raises error.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Dict
import pytest
import torch

from scripts.launch_a30_production import launch_a30_deployment
from memory_study_v2.backbones import MLPAnnual


@pytest.fixture
def mock_repo_env(tmp_path: Path):
    """Set up valid mock environment with repo config, manifests, and fold files."""
    manifest_dir = tmp_path / "rebuild_plan"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Mock config with production_authorized: false
    config_path = manifest_dir / "config.proposed.json"
    config_path.write_text(
        json.dumps({
            "production_authorized": False,
            "experiment_name": "mock_production",
            "neural": {
                "seeds": [7],
                "architectures": ["MLP_MOCK"],
            },
        }),
        encoding="utf-8",
    )

    # Mock manifests
    (manifest_dir / "universe_request.csv").write_text("project_security_id\ntest\n", encoding="utf-8")
    (manifest_dir / "fold_dimensions_manifest.json").write_text(json.dumps({"folds": {"2020": {}}}), encoding="utf-8")

    # Mock sample ID manifests for all 6 folds
    sample_ids_dir = manifest_dir / "sample_ids"
    sample_ids_dir.mkdir(parents=True, exist_ok=True)
    for year in range(2020, 2026):
        (sample_ids_dir / f"fold_{year}_sample_ids.json").write_text("{}", encoding="utf-8")

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "config_path": config_path,
        "sample_ids_dir": sample_ids_dir,
        "output_dir": output_dir,
        "manifest_dir": manifest_dir,
    }


def test_authorized_launch_calls_driver_and_preserves_repo_guard(mock_repo_env):
    """Authorized launch invokes driver_fn, provides runtime authorization, and keeps file guard."""
    calls = []

    def mock_driver(context: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(context)
        assert context["config"]["production_authorized"] is True
        return {
            "status": "PRODUCTION_SUCCESS",
            "jobs_started": ["fold_2020_job1"],
            "jobs_completed": ["fold_2020_job1"],
            "jobs_failed": [],
        }

    receipt = launch_a30_deployment(
        config_path=mock_repo_env["config_path"],
        output_dir=mock_repo_env["output_dir"],
        sample_ids_dir=mock_repo_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        driver_fn=mock_driver,
    )

    # Driver must be called exactly once
    assert len(calls) == 1
    assert receipt["status"] == "PREFLIGHT_PASS"
    assert receipt["execution"]["executed"] is True
    assert receipt["execution"]["driver_result"]["status"] == "PRODUCTION_SUCCESS"

    # Repository config file MUST remain production_authorized: false
    with open(mock_repo_env["config_path"], "r", encoding="utf-8") as f:
        disk_cfg = json.load(f)
    assert disk_cfg["production_authorized"] is False


def test_missing_storage_manifest_fails_closed_without_calling_driver(mock_repo_env):
    """Missing required manifest must raise RuntimeError and never execute the driver."""
    # Delete one of the required manifests
    missing_manifest = mock_repo_env["sample_ids_dir"] / "fold_2024_sample_ids.json"
    missing_manifest.unlink()

    calls = []

    def mock_driver(context: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(context)
        return {"status": "PRODUCTION_SUCCESS"}

    with pytest.raises(RuntimeError, match="Storage preflight check failed"):
        launch_a30_deployment(
            config_path=mock_repo_env["config_path"],
            output_dir=mock_repo_env["output_dir"],
            sample_ids_dir=mock_repo_env["sample_ids_dir"],
            authorize_production=True,
            check_only=False,
            driver_fn=mock_driver,
        )

    # Driver must never have been called
    assert len(calls) == 0

    # Receipt written on disk must record PREFLIGHT_STORAGE_ERROR
    receipt_file = mock_repo_env["output_dir"] / "a30_launch_receipt.json"
    assert receipt_file.exists()
    with open(receipt_file, "r", encoding="utf-8") as f:
        receipt_data = json.load(f)
    assert receipt_data["status"] == "PREFLIGHT_STORAGE_ERROR"
    assert any("sample ID manifests" in err for err in receipt_data["storage_errors"])


def test_driver_failure_propagates_and_records_production_failed(mock_repo_env):
    """If the production driver raises an exception, receipt must record PRODUCTION_FAILED."""
    def failing_driver(context: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("CUDA Out of Memory during training")

    with pytest.raises(RuntimeError, match="CUDA Out of Memory"):
        launch_a30_deployment(
            config_path=mock_repo_env["config_path"],
            output_dir=mock_repo_env["output_dir"],
            sample_ids_dir=mock_repo_env["sample_ids_dir"],
            authorize_production=True,
            check_only=False,
            driver_fn=failing_driver,
        )

    receipt_file = mock_repo_env["output_dir"] / "a30_launch_receipt.json"
    assert receipt_file.exists()
    with open(receipt_file, "r", encoding="utf-8") as f:
        receipt_data = json.load(f)
    assert receipt_data["status"] == "PRODUCTION_FAILED"
    assert "CUDA Out of Memory" in receipt_data["execution"]["error"]


def test_unauthorized_launch_does_not_execute_driver(mock_repo_env):
    """Calling without --authorize-production must lock execution and never call driver."""
    calls = []

    def mock_driver(context: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(context)
        return {"status": "PRODUCTION_SUCCESS"}

    receipt = launch_a30_deployment(
        config_path=mock_repo_env["config_path"],
        output_dir=mock_repo_env["output_dir"],
        sample_ids_dir=mock_repo_env["sample_ids_dir"],
        authorize_production=False,
        check_only=False,
        driver_fn=mock_driver,
    )

    assert len(calls) == 0
    assert receipt["status"] == "LOCKED_AWAITING_AUTHORIZATION"
    assert receipt["execution"]["executed"] is False
    assert receipt["authorization"]["can_proceed_to_production"] is False


def test_check_only_mode(mock_repo_env):
    """Check-only mode completes preflight checks without calling driver."""
    calls = []

    def mock_driver(context: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(context)
        return {"status": "PRODUCTION_SUCCESS"}

    receipt = launch_a30_deployment(
        config_path=mock_repo_env["config_path"],
        output_dir=mock_repo_env["output_dir"],
        sample_ids_dir=mock_repo_env["sample_ids_dir"],
        authorize_production=False,
        check_only=True,
        driver_fn=mock_driver,
    )

    assert len(calls) == 0
    assert receipt["status"] == "PREFLIGHT_PASS"
    assert receipt["execution"]["mode"] == "CHECK_ONLY"
    assert receipt["execution"]["executed"] is False


def test_corrupted_repo_config_with_authorized_true_fails(mock_repo_env):
    """If repository config has production_authorized: true, launcher must fail immediately."""
    # Corrupt repository configuration
    mock_repo_env["config_path"].write_text(
        json.dumps({"production_authorized": True}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Invariant violated"):
        launch_a30_deployment(
            config_path=mock_repo_env["config_path"],
            output_dir=mock_repo_env["output_dir"],
            sample_ids_dir=mock_repo_env["sample_ids_dir"],
            authorize_production=True,
            check_only=False,
        )


@pytest.fixture
def real_driver_env(tmp_path: Path):
    """Set up environment for real default driver execution tests."""
    manifest_dir = tmp_path / "rebuild_plan"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    config_path = manifest_dir / "config.proposed.json"
    config_path.write_text(
        json.dumps({
            "production_authorized": False,
            "experiment_name": "real_driver_test",
            "folds": [
                {
                    "evaluation_year": 2020,
                    "train_origin_start": "2015-01-01",
                    "training_availability_cutoff": "2017-12-31",
                    "validation_query_start": "2018-01-01",
                    "validation_query_end": "2018-12-31",
                    "development_query_start": "2019-01-01",
                    "development_query_end": "2019-12-31",
                },
                {"evaluation_year": 2021},
                {"evaluation_year": 2022},
                {"evaluation_year": 2023},
                {"evaluation_year": 2024},
                {"evaluation_year": 2025},
            ],
            "neural": {
                "seeds": [7],
                "architectures": ["MLP_ANNUAL_966_64_128_1"],
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "betas": [0.9, 0.999],
                "epsilon": 1e-08,
                "effective_batch": 32,
                "max_epochs": 2,
                "min_epochs": 1,
                "early_stop_patience": 5,
                "minimum_improvement": 1e-06,
                "gradient_norm_clip": 1.0,
                "optimizer": "AdamW",
                "transformer_dropout": 0.1,
            },
        }),
        encoding="utf-8",
    )

    (manifest_dir / "universe_request.csv").write_text("project_security_id\ntest\n", encoding="utf-8")
    (manifest_dir / "fold_dimensions_manifest.json").write_text(json.dumps({"folds": {"2020": {}}}), encoding="utf-8")

    sample_ids_dir = manifest_dir / "sample_ids"
    sample_ids_dir.mkdir(parents=True, exist_ok=True)
    for year in range(2020, 2026):
        (sample_ids_dir / f"fold_{year}_sample_ids.json").write_text("{}", encoding="utf-8")

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("FINAL_SUBMISSION_PACKAGE/data/cache/ohlcv")

    return {
        "config_path": config_path,
        "sample_ids_dir": sample_ids_dir,
        "output_dir": output_dir,
        "data_dir": data_dir,
    }


def test_default_driver_real_execution_and_downstream_artifacts(real_driver_env):
    """Default production driver executes real training, updates weights, and emits downstream artifacts."""
    receipt = launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=real_driver_env["output_dir"],
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,  # Exercises real default driver
        driver_kwargs={
            "min_epochs": 1,
            "max_epochs": 2,
            "micro_batch_size": 16,
            "effective_batch_size": 32,
        },
    )

    assert receipt["status"] == "PREFLIGHT_PASS"
    drv_res = receipt["execution"]["driver_result"]
    assert drv_res["status"] == "PRODUCTION_SUCCESS"
    assert drv_res["study_scope"] == "EXPLICIT_SUBSET"
    assert drv_res["executed_folds"] == [2020]
    assert len(drv_res["jobs_completed"]) == 1

    chk_dir = real_driver_env["output_dir"] / "fold_2020" / "checkpoints_MLP_ANNUAL_966_64_128_1_seed7"
    best_pt = chk_dir / "best_checkpoint.pt"
    assert best_pt.exists(), "best_checkpoint.pt must exist"

    st = torch.load(best_pt, map_location="cpu", weights_only=False)
    assert hasattr(st, "model_state")

    # Verify optimizer updates occurred (trained weights differ from initialization)
    initial_model = MLPAnnual(seed=7)
    initial_p = next(initial_model.parameters())
    trained_p = st.model_state["fc1.weight"]
    diff = float(torch.norm(initial_p - trained_p))
    assert diff > 1e-4, f"Optimizer must perform weight updates, got diff={diff}"

    # Verify downstream predictions and truthful job completion
    preds_file = chk_dir / "predictions.json"
    assert preds_file.exists()
    comp_file = chk_dir / "job_completion.json"
    assert comp_file.exists()
    with open(comp_file, "r", encoding="utf-8") as f:
        comp_data = json.load(f)
    assert comp_data["status"] == "STAGE_COMPLETED"
    assert "best_checkpoint" in comp_data["artifacts"]
    assert comp_data["artifacts"]["best_checkpoint"]["sha256"] == hashlib.sha256(best_pt.read_bytes()).hexdigest()

    # Verify downstream fold summary
    fold_summary = real_driver_env["output_dir"] / "fold_2020" / "fold_summary.json"
    assert fold_summary.exists()
    with open(fold_summary, "r", encoding="utf-8") as f:
        f_summary_data = json.load(f)
    assert f_summary_data["status"] == "FOLD_COMPLETED"
    assert f_summary_data["eval_queries_count"] > 0
    assert f_summary_data["bank_records_count"] > 0

    # Verify downstream release pipeline artifacts
    pipeline_summary_path = real_driver_env["output_dir"] / "pipeline_summary.json"
    assert pipeline_summary_path.exists()
    with open(pipeline_summary_path, "r", encoding="utf-8") as f:
        pipe_data = json.load(f)
    assert pipe_data["status"] == "PRODUCTION_SUCCESS"
    assert pipe_data["pipeline_stage"] == "RELEASE_VERIFIED"

    release_manifest_path = real_driver_env["output_dir"] / "release_manifest.json"
    assert release_manifest_path.exists()
    with open(release_manifest_path, "r", encoding="utf-8") as f:
        rel_data = json.load(f)
    assert "artifacts" in rel_data
    assert "pipeline_summary.json" in rel_data["artifacts"]

    # Verify accounts and ledger evidence
    accounts_dir = real_driver_env["output_dir"] / "accounts"
    assert accounts_dir.exists()
    sample_acct_dir = next(accounts_dir.iterdir())
    assert (sample_acct_dir / "decisions.json").exists()
    assert (sample_acct_dir / "trades.json").exists()
    assert (sample_acct_dir / "daily_nav.json").exists()

    # Verify analysis bundle and replay verification
    analysis_dir = real_driver_env["output_dir"] / "release_analysis"
    assert analysis_dir.exists()
    replay_report_path = analysis_dir / "replay_report.json"
    assert replay_report_path.exists()
    with open(replay_report_path, "r", encoding="utf-8") as f:
        rep_data = json.load(f)
    assert rep_data["status"] == "REPLAY_VERIFIED"


def test_default_driver_interrupted_job_resumes_rather_than_skips(real_driver_env):
    """An interrupted job without job_completion.json resumes from last_checkpoint.pt."""
    # Step 1: Run with an intentional interruption at macro step 1
    launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=real_driver_env["output_dir"],
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={
            "min_epochs": 1,
            "max_epochs": 2,
            "micro_batch_size": 16,
            "effective_batch_size": 32,
            "interrupt_at_macro_step": 1,
        },
    )

    chk_dir = real_driver_env["output_dir"] / "fold_2020" / "checkpoints_MLP_ANNUAL_966_64_128_1_seed7"
    last_pt = chk_dir / "last_checkpoint.pt"
    comp_file = chk_dir / "job_completion.json"
    assert last_pt.exists(), "Interrupted run must produce last_checkpoint.pt"
    assert not comp_file.exists(), "Interrupted run must NOT produce job_completion.json"

    # Step 2: Resume execution without interruption
    receipt_resumed = launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=real_driver_env["output_dir"],
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={
            "min_epochs": 1,
            "max_epochs": 2,
            "micro_batch_size": 16,
            "effective_batch_size": 32,
        },
    )

    drv_res = receipt_resumed["execution"]["driver_result"]
    assert drv_res["status"] == "PRODUCTION_SUCCESS"
    assert "fold_2020_MLP_ANNUAL_966_64_128_1_seed7" in drv_res["jobs_resumable"], (
        "Driver must detect existing last_checkpoint.pt as resumable"
    )
    assert "fold_2020_MLP_ANNUAL_966_64_128_1_seed7" in drv_res["jobs_completed"]
    assert comp_file.exists(), "Resumed run must complete and produce job_completion.json"


def test_default_driver_training_error_propagates_and_prevents_completion(real_driver_env):
    """A deliberate training error propagates failure, writes PRODUCTION_FAILED receipt, and omits completion."""
    chk_dir = real_driver_env["output_dir"] / "fold_2020" / "checkpoints_MLP_ANNUAL_966_64_128_1_seed7"

    # Supply an invalid micro_batch_size=0 to induce an immediate failure in the training loop
    with pytest.raises(Exception):
        launch_a30_deployment(
            config_path=real_driver_env["config_path"],
            data_dir=real_driver_env["data_dir"],
            output_dir=real_driver_env["output_dir"],
            sample_ids_dir=real_driver_env["sample_ids_dir"],
            authorize_production=True,
            check_only=False,
            selected_folds=[2020],
            selected_securities=["US_AAPL", "US_MSFT"],
            execution_mode="pilot",
            driver_fn=None,
            driver_kwargs={
                "micro_batch_size": 0,
            },
        )

    # Receipt must record PRODUCTION_FAILED
    receipt_file = real_driver_env["output_dir"] / "a30_launch_receipt.json"
    assert receipt_file.exists()
    with open(receipt_file, "r", encoding="utf-8") as f:
        receipt_data = json.load(f)
    assert receipt_data["status"] == "PRODUCTION_FAILED"
    assert receipt_data["execution"]["status"] == "PRODUCTION_FAILED"

    # Job completion marker must never have been created
    comp_file = chk_dir / "job_completion.json"
    assert not comp_file.exists(), "Failed job must NOT produce job_completion.json"


def test_prediction_to_trade_causality(real_driver_env):
    """Deterministic positive eligible prediction causes BUY fill; nonpositive/ineligible prevents fill."""
    # Step 1: Run with positive predictions -> produces BUY fills
    receipt_pos = launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=real_driver_env["output_dir"],
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={
            "min_epochs": 1,
            "max_epochs": 1,
            "micro_batch_size": 16,
            "effective_batch_size": 32,
            "custom_predictions": {"default": 0.05},
        },
    )
    assert receipt_pos["status"] == "PREFLIGHT_PASS"
    accts_dir = real_driver_env["output_dir"] / "accounts"
    sample_acct_dir = next(accts_dir.iterdir())
    with open(sample_acct_dir / "trades.json", "r", encoding="utf-8") as f:
        trades_pos = json.load(f)
    assert len(trades_pos) > 0, "Positive predictions must produce trade fills"
    assert any(tr["side"] == "BUY" for tr in trades_pos)

    # Step 2: In a separate output dir, run with negative predictions -> NO fills
    output_dir_neg = real_driver_env["output_dir"].parent / "output_neg"
    output_dir_neg.mkdir(parents=True, exist_ok=True)
    # Copy fold_2020 checkpoints so training doesn't have to rerun
    shutil.copytree(
        real_driver_env["output_dir"] / "fold_2020",
        output_dir_neg / "fold_2020",
    )
    # Remove old predictions from copied fold
    (output_dir_neg / "fold_2020" / "policy_predictions.json").unlink(missing_ok=True)
    (output_dir_neg / "fold_2020" / "predictions_manifest.json").unlink(missing_ok=True)

    receipt_neg = launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=output_dir_neg,
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={
            "custom_predictions": {"default": -0.05},
        },
    )
    assert receipt_neg["status"] == "PREFLIGHT_PASS"
    accts_dir_neg = output_dir_neg / "accounts"
    sample_acct_dir_neg = next(accts_dir_neg.iterdir())
    with open(sample_acct_dir_neg / "trades.json", "r", encoding="utf-8") as f:
        trades_neg = json.load(f)
    assert len(trades_neg) == 0, "Negative predictions must produce NO trade fills"


def test_legitimate_no_trade_path(real_driver_env):
    """When no eligible trades occur, account stays 100% cash, daily returns are 0, and NAV identity holds."""
    # Ensure trained checkpoints exist
    chk_dir = real_driver_env["output_dir"] / "fold_2020" / "checkpoints_MLP_ANNUAL_966_64_128_1_seed7"
    if not (chk_dir / "best_checkpoint.pt").exists():
        launch_a30_deployment(
            config_path=real_driver_env["config_path"],
            data_dir=real_driver_env["data_dir"],
            output_dir=real_driver_env["output_dir"],
            sample_ids_dir=real_driver_env["sample_ids_dir"],
            authorize_production=True,
            check_only=False,
            selected_folds=[2020],
            selected_securities=["US_AAPL", "US_MSFT"],
            execution_mode="pilot",
            driver_fn=None,
            driver_kwargs={"min_epochs": 1, "max_epochs": 1, "micro_batch_size": 16, "effective_batch_size": 32},
        )

    out_no_trade = real_driver_env["output_dir"].parent / "output_no_trade"
    out_no_trade.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        real_driver_env["output_dir"] / "fold_2020",
        out_no_trade / "fold_2020",
    )
    (out_no_trade / "fold_2020" / "policy_predictions.json").unlink(missing_ok=True)
    (out_no_trade / "fold_2020" / "predictions_manifest.json").unlink(missing_ok=True)

    receipt = launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=out_no_trade,
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={
            "custom_predictions": {"default": -0.05},
        },
    )
    assert receipt["status"] == "PREFLIGHT_PASS"
    drv_res = receipt["execution"]["driver_result"]
    assert drv_res["status"] == "PRODUCTION_SUCCESS"

    accts_dir = out_no_trade / "accounts"
    sample_acct_dir = next(accts_dir.iterdir())
    with open(sample_acct_dir / "trades.json", "r", encoding="utf-8") as f:
        trades = json.load(f)
    assert len(trades) == 0, "Cash-only ledger must have 0 trades"

    with open(sample_acct_dir / "daily_nav.json", "r", encoding="utf-8") as f:
        nav_history = json.load(f)
    assert len(nav_history) > 0

    init_capital = 100000.0
    compound_nav = init_capital
    for st in nav_history:
        # NAV identity: NAV == cash + holdings == cash == 100000.0
        assert abs(st["total_nav"] - (st["cash"] + st["holdings_value"])) < 1e-6
        assert abs(st["cash"] - init_capital) < 1e-6
        assert abs(st["holdings_value"] - 0.0) < 1e-6
        assert abs(st["daily_return"] - 0.0) < 1e-9
        compound_nav *= (1.0 + st["daily_return"])

    # Compound return reconciliation (within 1e-4)
    assert abs(compound_nav - init_capital) < 1e-4
    # Final cash reconciliation (within 1e-6)
    assert abs(nav_history[-1]["cash"] - init_capital) < 1e-6


def test_annual_continuity_multi_fold(real_driver_env):
    """2-fold continuous execution preserves portfolio state across fold boundary and liquidates only at end."""
    out_multi = real_driver_env["output_dir"].parent / "output_multi"
    out_multi.mkdir(parents=True, exist_ok=True)

    receipt = launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=out_multi,
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020, 2021],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={
            "min_epochs": 1,
            "max_epochs": 1,
            "micro_batch_size": 16,
            "effective_batch_size": 32,
        },
    )

    assert receipt["status"] == "PREFLIGHT_PASS"
    drv_res = receipt["execution"]["driver_result"]
    assert drv_res["status"] == "PRODUCTION_SUCCESS"
    assert drv_res["executed_folds"] == [2020, 2021]

    # Verify both folds have distinct checkpoints and models
    chk_2020 = out_multi / "fold_2020" / "checkpoints_MLP_ANNUAL_966_64_128_1_seed7" / "best_checkpoint.pt"
    chk_2021 = out_multi / "fold_2021" / "checkpoints_MLP_ANNUAL_966_64_128_1_seed7" / "best_checkpoint.pt"
    assert chk_2020.exists()
    assert chk_2021.exists()

    # Verify accounts continuity across years
    accts_dir = out_multi / "accounts"
    sample_acct_dir = next(accts_dir.iterdir())
    with open(sample_acct_dir / "daily_nav.json", "r", encoding="utf-8") as f:
        nav_history = json.load(f)

    sessions = [st["session"] for st in nav_history]
    has_2020 = any(s.startswith("2020") for s in sessions)
    has_2021 = any(s.startswith("2021") for s in sessions)
    assert has_2020 and has_2021, "Daily history must span both 2020 and 2021 continuous sessions"

    # Verify continuous transition: session dates are monotonically sorted
    assert sessions == sorted(sessions)

    # Verify terminal liquidation occurred only at the end of 2021 (last state holdings == 0.0)
    final_st = nav_history[-1]
    assert final_st["session"].startswith("2021")
    assert abs(final_st["holdings_value"]) < 1e-6
    assert abs(final_st["total_nav"] - final_st["cash"]) < 1e-6


def test_downstream_failure_propagates_and_records_failed(real_driver_env):
    """Induced downstream error propagates failure and writes PRODUCTION_FAILED receipt."""
    out_fail = real_driver_env["output_dir"].parent / "output_fail"
    out_fail.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="Induced downstream portfolio error"):
        launch_a30_deployment(
            config_path=real_driver_env["config_path"],
            data_dir=real_driver_env["data_dir"],
            output_dir=out_fail,
            sample_ids_dir=real_driver_env["sample_ids_dir"],
            authorize_production=True,
            check_only=False,
            selected_folds=[2020],
            selected_securities=["US_AAPL", "US_MSFT"],
            execution_mode="pilot",
            driver_fn=None,
            driver_kwargs={
                "min_epochs": 1,
                "max_epochs": 1,
                "micro_batch_size": 16,
                "effective_batch_size": 32,
                "induce_downstream_error": True,
            },
        )

    receipt_path = out_fail / "a30_launch_receipt.json"
    assert receipt_path.exists()
    with open(receipt_path, "r", encoding="utf-8") as f:
        receipt_data = json.load(f)
    assert receipt_data["status"] == "PRODUCTION_FAILED"
    assert receipt_data["execution"]["status"] == "PRODUCTION_FAILED"
    assert "Induced downstream portfolio error" in receipt_data["execution"]["error"]

    # Pipeline summary must NOT be created as RELEASE_VERIFIED
    assert not (out_fail / "pipeline_summary.json").exists()


def test_restart_after_downstream_failure_reuses_training(real_driver_env):
    """After downstream failure, restart detects existing job completion and reuses training without retraining."""
    out_restart = real_driver_env["output_dir"].parent / "output_restart"
    out_restart.mkdir(parents=True, exist_ok=True)

    # Step 1: Run with induced downstream error -> fails during portfolio simulation
    with pytest.raises(RuntimeError):
        launch_a30_deployment(
            config_path=real_driver_env["config_path"],
            data_dir=real_driver_env["data_dir"],
            output_dir=out_restart,
            sample_ids_dir=real_driver_env["sample_ids_dir"],
            authorize_production=True,
            check_only=False,
            selected_folds=[2020],
            selected_securities=["US_AAPL", "US_MSFT"],
            execution_mode="pilot",
            driver_fn=None,
            driver_kwargs={
                "min_epochs": 1,
                "max_epochs": 1,
                "micro_batch_size": 16,
                "effective_batch_size": 32,
                "induce_downstream_error": True,
            },
        )

    chk_dir = out_restart / "fold_2020" / "checkpoints_MLP_ANNUAL_966_64_128_1_seed7"
    comp_file = chk_dir / "job_completion.json"
    assert comp_file.exists(), "Training stage must have completed before downstream error"
    comp_mtime_1 = comp_file.stat().st_mtime

    # Step 2: Restart without error -> driver reuses training and completes downstream
    receipt = launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=out_restart,
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={
            "min_epochs": 1,
            "max_epochs": 1,
            "micro_batch_size": 16,
            "effective_batch_size": 32,
        },
    )

    assert receipt["status"] == "PREFLIGHT_PASS"
    drv_res = receipt["execution"]["driver_result"]
    assert drv_res["status"] == "PRODUCTION_SUCCESS"
    # Verify training was reused (job_completion.json mtime unchanged)
    assert comp_file.stat().st_mtime == comp_mtime_1

    # Verify downstream completed cleanly
    pipe_path = out_restart / "pipeline_summary.json"
    assert pipe_path.exists()
    with open(pipe_path, "r", encoding="utf-8") as f:
        pipe_data = json.load(f)
    assert pipe_data["status"] == "PRODUCTION_SUCCESS"
    assert pipe_data["pipeline_stage"] == "RELEASE_VERIFIED"


def test_corrupted_prediction_artifact_rejects_completion(real_driver_env):
    """Corrupted cached prediction record or sha256 mismatch rejects completion when strict check enabled."""
    out_corrupt = real_driver_env["output_dir"].parent / "output_corrupt"
    out_corrupt.mkdir(parents=True, exist_ok=True)

    # First, run a successful run to generate authentic prediction artifacts
    launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=out_corrupt,
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={"min_epochs": 1, "max_epochs": 1, "micro_batch_size": 16, "effective_batch_size": 32},
    )

    # Tamper with policy_predictions.json to create a sha256 mismatch
    pred_path = out_corrupt / "fold_2020" / "policy_predictions.json"
    assert pred_path.exists()
    pred_path.write_text("[{\"corrupted\": true}]", encoding="utf-8")
    (out_corrupt / "pipeline_completion.json").unlink(missing_ok=True)
    (out_corrupt / "pipeline_summary.json").unlink(missing_ok=True)

    # Run with fail_on_corrupted_predictions=True -> must fail closed with RuntimeError
    with pytest.raises(RuntimeError, match="Corrupted predictions artifact rejected"):
        launch_a30_deployment(
            config_path=real_driver_env["config_path"],
            data_dir=real_driver_env["data_dir"],
            output_dir=out_corrupt,
            sample_ids_dir=real_driver_env["sample_ids_dir"],
            authorize_production=True,
            check_only=False,
            selected_folds=[2020],
            selected_securities=["US_AAPL", "US_MSFT"],
            execution_mode="pilot",
            driver_fn=None,
            driver_kwargs={
                "fail_on_corrupted_predictions": True,
            },
        )

def test_multi_market_calendar_holiday_isolation(tmp_path):
    """Two market calendars with different holidays: closed market does not age positions or record observations."""
    from memory_study_v2.venue_calendar import VenueCalendar
    from scripts.launch_a30_production import run_continuous_portfolio_simulation, PolicyPredictionRecord
    import pandas as pd
    import numpy as np

    # M1 open all 4 sessions, M2 closed on 2020-01-06 (holiday)
    m1_sessions = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    m2_sessions = ["2020-01-02", "2020-01-03", "2020-01-07"]
    cal1 = VenueCalendar(sessions=m1_sessions)
    cal2 = VenueCalendar(sessions=m2_sessions)

    # Build synthetic sec_info for M1_SEC1 and M2_SEC2
    def _make_sec_data(sec_id, market, sessions, price):
        tr_df = pd.DataFrame({
            "session": sessions,
            "raw_open": [price] * len(sessions),
            "raw_close": [price] * len(sessions),
        })
        val_df = pd.DataFrame({
            "session": sessions,
            "bar_status": ["VALID"] * len(sessions),
        })
        feats_df = pd.DataFrame({
            "session": sessions,
            "volatility_21": [0.01] * len(sessions),
            "atr_ratio_14": [0.01] * len(sessions),
            "momentum_21": [0.05] * len(sessions),
        })
        return {
            "market": market,
            "calendar": cal1 if market == "M1" else cal2,
            "tr_df": tr_df,
            "val_df": val_df,
            "feats_df": feats_df,
            "sess_to_row": {s: i for i, s in enumerate(sessions)},
        }

    sec_info = {
        "M1_SEC1": _make_sec_data("M1_SEC1", "M1", m1_sessions, 100.0),
        "M2_SEC2": _make_sec_data("M2_SEC2", "M2", m2_sessions, 50.0),
    }

    fold_data_by_year = {
        2020: {
            "sec_info": sec_info,
            "market_calendars": {"M1": cal1, "M2": cal2},
            "eval_records": [],
        }
    }

    # Queue entry predictions on session 0
    preds = [
        PolicyPredictionRecord(
            query_id="2020_M1_SEC1_2020-01-02",
            security_id="M1_SEC1",
            market="M1",
            decision_session="2020-01-02",
            fold=2020,
            policy_id="MLP_BASE",
            realization_id=7,
            prediction=0.05,
            source_artifact_id="art1",
        ),
        PolicyPredictionRecord(
            query_id="2020_M2_SEC2_2020-01-02",
            security_id="M2_SEC2",
            market="M2",
            decision_session="2020-01-02",
            fold=2020,
            policy_id="MLP_BASE",
            realization_id=7,
            prediction=0.05,
            source_artifact_id="art2",
        ),
    ]

    out_dir = tmp_path / "multi_mkt_sim"
    res = run_continuous_portfolio_simulation(
        folds_to_run=[2020],
        fold_data_by_year=fold_data_by_year,
        predictions_by_year={2020: preds},
        output_dir=out_dir,
    )

    accts = res["policy_accounts"]
    acct1 = accts[("MLP_BASE", "M1", 7)]
    acct2 = accts[("MLP_BASE", "M2", 7)]

    # M1 was open for 4 sessions
    assert len(acct1.daily_history) == 4
    m1_dates = [st.session for st in acct1.daily_history]
    assert m1_dates == ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]

    # M2 was open for 3 sessions (holiday on 2020-01-06)
    assert len(acct2.daily_history) == 3
    m2_dates = [st.session for st in acct2.daily_history]
    assert m2_dates == ["2020-01-02", "2020-01-03", "2020-01-07"]

    # On 2020-01-06, M2 did NOT step and did NOT record an observation
    assert "2020-01-06" not in m2_dates

    # Union calendar and market open mask verification
    union_dates = res["union_sessions"]
    assert union_dates == ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    open_mask = res["market_open_mask"]

    # Account 1 (M1) mask: True on all 4 sessions
    idx1 = res["sorted_account_keys"].index(("MLP_BASE", "M1", 7))
    assert list(open_mask[idx1]) == [True, True, True, True]

    # Account 2 (M2) mask: False on 2020-01-06 (index 2)
    idx2 = res["sorted_account_keys"].index(("MLP_BASE", "M2", 7))
    assert list(open_mask[idx2]) == [True, True, False, True]


def test_select_mixture_sr_development_ledger_scores():
    """select_mixture_sr uses dev_eval_fn to score against actual development portfolio Sharpe ratios."""
    from memory_study_v2.integration import select_mixture_sr
    import numpy as np

    # Synthetic dev predictions for 2 seeds in 1 market
    dev_base = {
        7: np.array([0.01, -0.02, 0.03, -0.01]),
        17: np.array([0.02, -0.01, 0.02, -0.02]),
    }
    dev_mem = np.array([0.05, 0.06, 0.04, 0.05])
    dev_y = np.array([0.05, 0.04, 0.03, 0.05])
    dev_mkts = np.array(["US", "US", "US", "US"])

    # Callback simulating development portfolio Sharpe
    # lam=0.0 -> poor Sharpe (-0.5), lam=1.0 -> excellent Sharpe (+2.5)
    def mock_dev_eval(lam: float, seed: int, market: str) -> float:
        if lam == 1.0:
            return 2.5
        elif lam == 0.5:
            return 1.2
        elif lam == 0.25:
            return 0.5
        elif lam == 0.10:
            return 0.1
        else:
            return -0.5

    sel = select_mixture_sr(
        dev_base_by_seed=dev_base,
        dev_mem_preds=dev_mem,
        dev_targets=dev_y,
        dev_markets=dev_mkts,
        lambda_grid=[0.0, 0.10, 0.25, 0.50, 1.00],
        dev_eval_fn=mock_dev_eval,
    )

    assert sel.selected_lambda == 1.0
    assert sel.grid_scores[1.0] == 2.5
    assert sel.grid_scores[0.0] == -0.5
    assert sel.grid_scores[1.0] > sel.grid_scores[0.0]


def test_passive_benchmark_holds_beyond_max_holding_sessions():
    """PASSIVE_EQUAL_WEIGHT holds beyond active 63-session horizon without stops or age exits."""
    from memory_study_v2.execution import PassiveEqualWeightAccount, PortfolioAccount

    # Setup 70 sessions (> 63 max holding sessions)
    sessions = [f"2020-01-{i+1:02d}" for i in range(70)]

    active_acct = PortfolioAccount(
        initial_capital=100000.0,
        max_positions=3,
        max_holding_sessions=63,
    )
    passive_acct = PassiveEqualWeightAccount(
        initial_capital=100000.0,
        universe_size=2,
    )

    # Initialize entries at session 0
    passive_acct.initialize_passive_entries(["SEC_A", "SEC_B"])
    active_acct.pending_entries = ["SEC_A"]

    open_p = {"SEC_A": 100.0, "SEC_B": 100.0}
    trad_flags = {"SEC_A": True, "SEC_B": True}
    close_p = {"SEC_A": 100.0, "SEC_B": 100.0}
    atr = {"SEC_A": 0.01, "SEC_B": 0.01}

    # Step through all 70 sessions
    for idx, sess in enumerate(sessions):
        is_terminal = (idx == len(sessions) - 1)

        # Open fills
        active_acct.process_open_fills(sess, open_p, trad_flags)
        passive_acct.process_open_fills(sess, open_p, trad_flags)

        # Close valuation & stop evaluation
        active_acct.evaluate_close_stops_and_update_state(sess, close_p, atr)
        passive_acct.evaluate_close_stops_and_update_state(sess, close_p, atr)

        if not is_terminal:
            active_acct.plan_entries_at_close([])
            passive_acct.plan_entries_at_close([])
        else:
            active_acct.execute_terminal_liquidation(sess, close_p)
            passive_acct.execute_terminal_liquidation(sess, close_p)

    # Passive held both securities through session 69 with NO MAX_AGE exit
    passive_trade_reasons = [t.reason for t in passive_acct.trades]
    assert "MAX_AGE" not in passive_trade_reasons
    assert "STOP" not in passive_trade_reasons
    assert passive_trade_reasons[0] == "PASSIVE_INITIAL_ENTRY"
    assert passive_trade_reasons[1] == "PASSIVE_INITIAL_ENTRY"
    assert "TERMINAL" in passive_trade_reasons

    # Active queued MAX_AGE exit on session 63 and sold on session 64
    active_trade_sides = [t.side for t in active_acct.trades]
    assert "SELL_MAX_AGE" in active_trade_sides


def test_cached_completion_invalidates_on_scope_or_config_change(real_driver_env, tmp_path):
    """Reusing an output directory with changed scope or config invalidates cached completion."""
    from scripts.launch_a30_production import launch_a30_deployment
    import json

    out_dir = tmp_path / "cache_binding_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # First run on US_AAPL only
    receipt1 = launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=out_dir,
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={
            "max_epochs": 1,
            "micro_batch_size": 16,
            "effective_batch_size": 32,
        },
    )
    assert receipt1["execution"]["status"] == "PRODUCTION_SUCCESS"

    comp_file = out_dir / "pipeline_completion.json"
    assert comp_file.exists()
    with open(comp_file, "r", encoding="utf-8") as f:
        comp_data = json.load(f)
    assert comp_data["run_identity"]["requested_scope"]["securities"] == ["US_AAPL"]

    # Second run reusing out_dir, but changing selected securities to US_AAPL + US_MSFT
    receipt2 = launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=out_dir,
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={
            "max_epochs": 1,
            "micro_batch_size": 16,
            "effective_batch_size": 32,
        },
    )

    # Must NOT have returned cached run; must have executed for new scope
    assert receipt2["execution"]["status"] == "PRODUCTION_SUCCESS"
    with open(comp_file, "r", encoding="utf-8") as f:
        comp_data2 = json.load(f)
    assert sorted(comp_data2["run_identity"]["requested_scope"]["securities"]) == ["US_AAPL", "US_MSFT"]


def test_production_fails_on_missing_market_calendar_without_fallback(real_driver_env, tmp_path):
    """Production mode strictly requires validated fixture calendar; fails closed without falling back to US or prices."""
    from memory_study_v2.venue_calendar import get_market_venue_calendar

    # 1. Non-existent / non-US market with no fixture calendar in production mode must fail closed
    with pytest.raises(ValueError, match="Missing authoritative venue calendar for market 'JP'"):
        get_market_venue_calendar("JP", execution_mode="production")

    # In pilot mode, permissive fallback is allowed but explicitly marked
    pilot_cal = get_market_venue_calendar("JP", execution_mode="pilot")
    assert pilot_cal.is_inferred is True
    assert pilot_cal.fallback_source == "us_calendar_substitute:JP"

    # 2. End-to-end launcher in production mode with unmapped market fails upfront before training
    out_dir = tmp_path / "prod_missing_cal"
    with pytest.raises(ValueError, match="Missing authoritative venue calendar for market 'JP'"):
        launch_a30_deployment(
            config_path=real_driver_env["config_path"],
            data_dir=real_driver_env["data_dir"],
            output_dir=out_dir,
            sample_ids_dir=real_driver_env["sample_ids_dir"],
            authorize_production=True,
            check_only=False,
            selected_folds=[2020],
            selected_securities=["JP_7203"],
            execution_mode="production",
        )


def test_pipeline_reuse_invalidates_on_checkpoint_or_code_change(real_driver_env, tmp_path):
    """Layer 1: Pipeline completion reuse is invalidated if checkpoint hash or code revision differs."""
    out_dir = tmp_path / "pipe_reuse_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initial complete run
    receipt1 = launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=out_dir,
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={
            "min_epochs": 1,
            "max_epochs": 1,
            "micro_batch_size": 16,
            "effective_batch_size": 32,
            "code_revision": "initial_code_rev_001",
        },
    )
    assert receipt1["execution"]["status"] == "PRODUCTION_SUCCESS"
    assert receipt1["execution"]["driver_result"].get("reused_cached_completion") is not True

    # Immediate rerun with identical revision and checkpoints -> reuses cached completion
    receipt_reused = launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=out_dir,
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={
            "min_epochs": 1,
            "max_epochs": 1,
            "micro_batch_size": 16,
            "effective_batch_size": 32,
            "code_revision": "initial_code_rev_001",
        },
    )
    assert receipt_reused["execution"]["driver_result"].get("reused_cached_completion") is True

    # 2. Invalidate via code revision change
    receipt_code_changed = launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=out_dir,
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={
            "min_epochs": 1,
            "max_epochs": 1,
            "micro_batch_size": 16,
            "effective_batch_size": 32,
            "code_revision": "modified_code_rev_002",
        },
    )
    assert receipt_code_changed["execution"]["status"] == "PRODUCTION_SUCCESS"
    assert receipt_code_changed["execution"]["driver_result"].get("reused_cached_completion") is not True

    # 3. Invalidate via checkpoint modification on disk
    best_pt = out_dir / "fold_2020" / "checkpoints_MLP_ANNUAL_966_64_128_1_seed7" / "best_checkpoint.pt"
    assert best_pt.exists()
    # Overwrite best_checkpoint with modified bytes
    orig_bytes = best_pt.read_bytes()
    best_pt.write_bytes(orig_bytes + b"tampered_bytes")

    receipt_chk_changed = launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=out_dir,
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={
            "min_epochs": 1,
            "max_epochs": 1,
            "micro_batch_size": 16,
            "effective_batch_size": 32,
            "code_revision": "modified_code_rev_002",
        },
    )
    assert receipt_chk_changed["execution"]["status"] == "PRODUCTION_SUCCESS"
    assert receipt_chk_changed["execution"]["driver_result"].get("reused_cached_completion") is not True


def test_training_cache_invalidates_on_inputs_or_config_change(real_driver_env, tmp_path):
    """Layer 2: Invalidating pipeline completion does not blindly reuse incompatible training checkpoint."""
    out_dir = tmp_path / "train_cache_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run with effective_batch_size=32
    receipt1 = launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=out_dir,
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={
            "min_epochs": 1,
            "max_epochs": 1,
            "micro_batch_size": 16,
            "effective_batch_size": 32,
        },
    )
    assert receipt1["execution"]["status"] == "PRODUCTION_SUCCESS"

    chk_dir = out_dir / "fold_2020" / "checkpoints_MLP_ANNUAL_966_64_128_1_seed7"
    job_comp_file = chk_dir / "job_completion.json"
    assert job_comp_file.exists()
    initial_comp = json.loads(job_comp_file.read_text(encoding="utf-8"))
    initial_ident_sha = initial_comp["training_identity"]["job_identity_sha256"]

    # 2. Invalidate pipeline completion
    (out_dir / "pipeline_completion.json").unlink(missing_ok=True)
    (out_dir / "pipeline_summary.json").unlink(missing_ok=True)

    # 3. Re-run with different training configuration (effective_batch_size=64)
    receipt2 = launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=out_dir,
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={
            "min_epochs": 1,
            "max_epochs": 1,
            "micro_batch_size": 16,
            "effective_batch_size": 64,
        },
    )
    assert receipt2["execution"]["status"] == "PRODUCTION_SUCCESS"

    # Verify Stage 1 detected identity mismatch and trained fresh
    new_comp = json.loads(job_comp_file.read_text(encoding="utf-8"))
    new_ident_sha = new_comp["training_identity"]["job_identity_sha256"]
    assert new_ident_sha != initial_ident_sha
    assert (
        new_comp["training_identity"]["neural_configuration_sha256"]
        != initial_comp["training_identity"]["neural_configuration_sha256"]
    )
    assert new_comp["artifacts"]["best_checkpoint"]["sha256"]


def test_checkpoint_modification_regenerates_policy_predictions(real_driver_env, tmp_path):
    """Modified model checkpoint triggers prediction_identity mismatch, regenerating forecasts from that model while preserving training."""
    import numpy as np
    from memory_study_v2.backbones import MLPAnnual
    from scripts.launch_a30_production import prepare_fold_data

    out_dir = tmp_path / "pred_cache_inval_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initial complete run
    receipt1 = launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=out_dir,
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={
            "min_epochs": 1,
            "max_epochs": 1,
            "micro_batch_size": 16,
            "effective_batch_size": 32,
        },
    )
    assert receipt1["execution"]["status"] == "PRODUCTION_SUCCESS"

    manifest_path = out_dir / "fold_2020" / "predictions_manifest.json"
    pred_path = out_dir / "fold_2020" / "policy_predictions.json"
    assert manifest_path.exists()
    assert pred_path.exists()

    orig_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "prediction_identity" in orig_manifest
    orig_pred_ident = orig_manifest["prediction_identity"]

    orig_preds = json.loads(pred_path.read_text(encoding="utf-8"))
    orig_mlp_preds = {
        r["query_id"]: r["prediction"]
        for r in orig_preds
        if r["policy_id"] == "MLP_BASE" and r["realization_id"] == 7
    }

    # 2. Modify model checkpoint so its forecasts demonstrably change, keeping query IDs fixed
    chk_dir = out_dir / "fold_2020" / "checkpoints_MLP_ANNUAL_966_64_128_1_seed7"
    best_pt = chk_dir / "best_checkpoint.pt"
    assert best_pt.exists()

    chk_st = torch.load(best_pt, map_location="cpu", weights_only=False)
    for k in chk_st.model_state:
        chk_st.model_state[k] = -chk_st.model_state[k] * 2.0
    torch.save(chk_st, best_pt)
    new_chk_sha = hashlib.sha256(best_pt.read_bytes()).hexdigest()

    # Update job_completion.json so Stage 1 preserves this training checkpoint
    job_comp_file = chk_dir / "job_completion.json"
    job_comp = json.loads(job_comp_file.read_text(encoding="utf-8"))
    job_comp["artifacts"]["best_checkpoint"]["sha256"] = new_chk_sha
    job_comp_file.write_text(json.dumps(job_comp), encoding="utf-8")
    chk_mtime_before = job_comp_file.stat().st_mtime

    # Record portfolio manifest mtime before restart so we can verify it is regenerated
    port_manifest_path = out_dir / "portfolio_manifest.json"
    port_manifest_mtime_before = port_manifest_path.stat().st_mtime if port_manifest_path.exists() else None

    # Invalidate pipeline completion to trigger restart
    (out_dir / "pipeline_summary.json").unlink(missing_ok=True)
    (out_dir / "pipeline_completion.json").unlink(missing_ok=True)

    # 3. Restart deployment
    receipt2 = launch_a30_deployment(
        config_path=real_driver_env["config_path"],
        data_dir=real_driver_env["data_dir"],
        output_dir=out_dir,
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
        driver_fn=None,
        driver_kwargs={
            "min_epochs": 1,
            "max_epochs": 1,
            "micro_batch_size": 16,
            "effective_batch_size": 32,
        },
    )
    assert receipt2["execution"]["status"] == "PRODUCTION_SUCCESS"

    # Verify Stage 1 preserved compatible training checkpoint (not retrained)
    assert job_comp_file.stat().st_mtime == chk_mtime_before
    assert best_pt.exists()

    # Verify predictions manifest updated with new prediction identity
    new_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    new_pred_ident = new_manifest["prediction_identity"]
    assert new_pred_ident["prediction_identity_sha256"] != orig_pred_ident["prediction_identity_sha256"]
    assert new_pred_ident["producing_checkpoints"] != orig_pred_ident["producing_checkpoints"]

    # Verify policy predictions demonstrably regenerated from modified model
    new_preds = json.loads(pred_path.read_text(encoding="utf-8"))
    new_mlp_preds = {
        r["query_id"]: r["prediction"]
        for r in new_preds
        if r["policy_id"] == "MLP_BASE" and r["realization_id"] == 7
    }
    assert len(new_mlp_preds) == len(orig_mlp_preds)
    assert new_mlp_preds != orig_mlp_preds
    assert any(abs(new_mlp_preds[qid] - orig_mlp_preds[qid]) > 1e-4 for qid in orig_mlp_preds)

    # Assert numerical identity: regenerated forecasts match evaluating the modified model directly
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modified_model = MLPAnnual(seed=7).to(device)
    modified_model.load_state_dict(chk_st.model_state)
    modified_model.eval()
    fold_data = prepare_fold_data(
        fold_year=2020,
        data_dir=real_driver_env["data_dir"],
        sample_ids_dir=real_driver_env["sample_ids_dir"],
        selected_securities=["US_AAPL", "US_MSFT"],
        execution_mode="pilot",
    )
    with torch.no_grad():
        expected_eval = modified_model(torch.tensor(fold_data["eval_x_mlp"], dtype=torch.float32, device=device)).squeeze(-1).cpu().numpy().tolist()
    actual_eval = [new_mlp_preds[r.query_id] for r in fold_data["eval_records"]]
    np.testing.assert_allclose(actual_eval, expected_eval, atol=1e-3)

    # Assert dependent portfolio outputs were regenerated, not served from a stale cache.
    # The portfolio manifest must exist and must have been rewritten during the second run.
    assert port_manifest_path.exists(), "portfolio_manifest.json must exist after second run"
    port_manifest_mtime_after = port_manifest_path.stat().st_mtime
    assert port_manifest_mtime_after != port_manifest_mtime_before, (
        "portfolio_manifest.json mtime unchanged: dependent portfolio outputs were NOT regenerated "
        "after prediction identity mismatch — stale artifacts were served."
    )


