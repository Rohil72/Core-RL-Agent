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


