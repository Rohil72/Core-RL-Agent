"""Contracts and schema validation for clean memory-system rebuild (v2).

Validates config.proposed.json, universe_request.csv, feature_contract.csv,
and acceptance_tests.csv against authoritative specification.
Enforces strict rejection of unknown keys, NaNs, missing fields, and unverified production gates.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


class ContractValidationError(Exception):
    """Raised when a configuration or data object violates the rebuild contract."""
    pass


def to_canonical_json(obj: Any) -> str:
    """Serialize object to deterministic canonical JSON (sorted keys, compact, UTF-8, no NaN)."""
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def validate_no_nan(obj: Any, path: str = "$") -> None:
    """Recursively check that no NaN or infinite floating point values exist."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            validate_no_nan(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            validate_no_nan(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        if math.isnan(obj):
            raise ContractValidationError(f"NaN value detected at {path}")
        if math.isinf(obj):
            raise ContractValidationError(f"Infinite value detected at {path}")


def _check_allowed_keys(d: Dict[str, Any], allowed_keys: Set[str], context: str) -> None:
    """Reject any keys not explicitly in the allowed set."""
    actual_keys = set(d.keys())
    extra = actual_keys - allowed_keys
    if extra:
        raise ContractValidationError(
            f"Unknown configuration keys in {context}: {sorted(extra)}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )


ROOT_ALLOWED_KEYS: Set[str] = {
    "schema_version",
    "created_date",
    "status",
    "production_authorized",
    "authoritative_document",
    "scope",
    "data",
    "folds",
    "representation",
    "neural",
    "ridge",
    "memory",
    "integration",
    "arms",
    "execution",
    "analysis",
    "cost_stress",
    "counts",
    "operations",
    "production_gates",
}

SCOPE_ALLOWED_KEYS: Set[str] = {
    "namespace",
    "preserve_legacy_release",
    "memory_design_is_contribution",
    "evaluation_status",
    "untouched_holdout_claim",
    "production_search_trials_per_backbone_fold_seed",
    "evaluation_2026_enabled",
}

DATA_ALLOWED_KEYS: Set[str] = {
    "candidate_provider",
    "provider_mode",
    "start_inclusive",
    "end_exclusive",
    "warmup_only_end",
    "label_tail_start",
    "label_tail_end_inclusive",
    "label_tail_role",
    "universe_manifest",
    "primary_count",
    "requested_count",
    "settings",
    "concurrent_requests",
    "max_attempts",
    "retry_wait_seconds",
    "production_price_basis",
    "automatic_price_basis_fallback",
    "calendar_basis",
    "point_in_time_universe_claim",
    "backfill_prices",
    "impute_unknown_labels",
    "factor_analysis_status",
}

DATA_SETTINGS_ALLOWED_KEYS: Set[str] = {
    "interval",
    "auto_adjust",
    "back_adjust",
    "repair",
    "actions",
    "keepna",
    "prepost",
}

FOLD_ALLOWED_KEYS: Set[str] = {
    "evaluation_year",
    "train_origin_start",
    "training_availability_cutoff",
    "validation_query_start",
    "validation_query_end",
    "validation_label_cutoff",
    "development_query_start",
    "development_query_end",
    "development_label_cutoff",
    "evaluation_query_start",
    "evaluation_query_end",
    "bank_cutoff",
}

REPRESENTATION_ALLOWED_KEYS: Set[str] = {
    "feature_contract",
    "features",
    "window_sessions",
    "patch_sessions",
    "patch_count",
    "flattened_dim",
    "exclude_decision_session",
    "feature_dtype",
    "statistics_dtype",
    "feature_standard_deviation_ddof",
    "scaler_ddof",
    "scaler_std_floor",
    "clip",
    "feature_denominator_guard",
    "minimum_feature_warmup_bars",
    "missing_feature_policy",
    "target_sessions",
    "target_basis",
    "target_units",
    "target_requires_uninterrupted_valid_segment",
}

NEURAL_ALLOWED_KEYS: Set[str] = {
    "seeds",
    "architectures",
    "fresh_initialization",
    "legacy_checkpoint_loading",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "betas",
    "epsilon",
    "effective_batch",
    "max_epochs",
    "min_epochs",
    "early_stop_patience",
    "minimum_improvement",
    "selection",
    "training_objective",
    "gradient_norm_clip",
    "scheduler",
    "amp",
    "tf32",
    "deterministic_algorithms",
    "layernorm_epsilon",
    "transformer_dropout",
}

RIDGE_ALLOWED_KEYS: Set[str] = {
    "l2_coefficient_on_equal_market_mse",
    "penalize_intercept",
    "dtype",
}

MEMORY_ALLOWED_KEYS: Set[str] = {
    "k",
    "maturity_sessions",
    "refresh",
    "intra_year_refresh",
    "distance",
    "reference_dtype",
    "same_canonical_security_excluded",
    "issuer_wide_isolation_claim",
    "max_records_per_security",
    "minimum_origin_spacing_sessions",
    "candidate_buffer_start",
    "candidate_expansion_factor",
    "candidate_exhaustion",
    "insufficient_neighbors",
    "tie_break",
    "random_master_seeds",
    "random_generator",
    "development_only_k_sensitivity",
    "plain_knn_removes",
    "production_backend",
    "query_chunk_rows",
    "bank_chunk_rows",
    "memory_valid_segment_sessions_after_origin",
    "proposal_absolute_error_budget",
    "proposal_boundary_margin",
    "final_distances",
    "approximate_nearest_neighbor_index",
}

INTEGRATION_ALLOWED_KEYS: Set[str] = {
    "gate_diagnostics",
    "gate_scale_floor",
    "gate_optimizer",
    "gate_learning_rate",
    "gate_weight_decay",
    "gate_betas",
    "gate_epsilon",
    "gate_steps",
    "gate_initial_weights",
    "gate_initial_bias",
    "gate_objective",
    "mixture_grid",
    "mixture_objectives",
    "mixture_tie_tolerance",
    "mixture_tie_break",
}

ARM_ALLOWED_KEYS: Set[str] = {
    "id",
    "realizations_per_market",
}

EXECUTION_ALLOWED_KEYS: Set[str] = {
    "initial_capital_account_units",
    "max_positions",
    "primary_path",
    "annual_account_reset",
    "development_account_reset",
    "cash_budget_fraction",
    "entry_lot",
    "commission_per_side",
    "slippage_per_side",
    "volatility_denominator_guard",
    "allow_nonpositive_entry_score",
    "fill",
    "exit_before_entry",
    "pending_exit_frees_slot_at_signal_time",
    "entry_age",
    "maximum_holding_sessions",
    "minimum_stop_fraction",
    "atr_stop_multiplier",
    "stale_held_session_review_threshold",
    "dividend_cash_timing",
    "terminal_entry_blackout_sessions",
    "terminal_fill",
    "quote_unit_default",
    "allow_negative_cash",
    "production_ledger_reconciliation_tolerance_account_units",
}

ANALYSIS_ALLOWED_KEYS: Set[str] = {
    "annualization",
    "portfolio_standard_deviation_ddof",
    "sharpe_zero_std_threshold",
    "sharpe_risk_free_subtraction",
    "headline",
    "bootstrap_draws",
    "bootstrap_primary_weeks",
    "bootstrap_sensitivity_weeks",
    "bootstrap_stratification",
    "bootstrap_generator",
    "bootstrap_quantile_method",
    "primary_contrasts",
    "multiplicity",
    "variance_roundoff_tolerance",
    "analysis_replay_tolerance",
    "daily_rank_ic_minimum_assets",
}

CONTRAST_ALLOWED_KEYS: Set[str] = {
    "id",
    "candidate",
    "comparator",
}

COST_STRESS_ALLOWED_KEYS: Set[str] = {
    "arms",
    "slippage_per_side",
    "commission_per_side",
    "reexecute_ledgers",
}

COUNTS_ALLOWED_KEYS: Set[str] = {
    "folds",
    "markets",
    "neural_selected_fits",
    "gate_fits",
    "ridge_fits",
    "configurations",
    "logical_realizations_per_market",
    "primary_continuous_paths",
    "primary_logical_market_year_cells",
    "cost_stress_additional_continuous_paths",
}

OPERATIONS_ALLOWED_KEYS: Set[str] = {
    "laptop",
    "a30",
    "vm_max_allocation_hours",
    "compute_safety_multiplier",
    "minimum_export_verification_hours",
    "contingency_hours",
    "maximum_backup_interval_minutes",
    "pilot_training_epochs",
    "pilot_warmup_optimizer_steps",
    "pilot_timed_optimizer_steps",
    "pilot_retrieval_queries",
    "pilot_bootstrap_draws",
}

OPERATIONS_PROFILE_ALLOWED_KEYS: Set[str] = {
    "microbatch",
    "loader_workers",
    "application_ram_ceiling_decimal_GB",
    "gpu_allocation_ceiling_decimal_GB",
    "working_artifact_ceiling_decimal_GB",
    "minimum_free_disk_decimal_GB",
}

PRODUCTION_GATES_ALLOWED_KEYS: Set[str] = {
    "reviewed_full_code_commit",
    "frozen_data_hashes",
    "provider_price_and_action_basis_verified",
    "quote_units_and_venue_calendars_verified",
    "exceptional_actions_and_delistings_resolved",
    "architecture_lineage_approved",
    "data_access_and_redistribution_scope_recorded",
    "all_acceptance_tests_passed",
    "measured_runtime_and_resources_accepted",
    "vm_actual_expiry_utc",
    "vm_persistence_confirmed",
    "off_vm_backup_destination_verified",
    "user_freeze_approval",
}

EXPECTED_16_ARMS: List[tuple[str, int]] = [
    ("MEM_SIM", 1),
    ("KNN_PLAIN", 1),
    ("MEM_RANDOM", 3),
    ("HIST_PRIOR", 1),
    ("RIDGE_ANNUAL", 1),
    ("MLP_BASE", 3),
    ("TRANS_BASE", 3),
    ("MLP_MIX_MSE", 3),
    ("TRANS_MIX_MSE", 3),
    ("MLP_MIX_SR", 3),
    ("TRANS_MIX_SR", 3),
    ("MLP_GATE", 3),
    ("TRANS_GATE", 3),
    ("MOMENTUM_21", 1),
    ("VOL_MOMENTUM_21", 1),
    ("PASSIVE_EQUAL_WEIGHT", 1),
]

EXPECTED_FEATURES_ORDERED: List[str] = [
    "return_1",
    "momentum_3",
    "momentum_10",
    "momentum_21",
    "volatility_21",
    "volume_mean_21",
    "volume_ratio_21",
    "volume_change_1",
    "intraday_range",
    "drawdown_252",
    "trend_slope_21",
    "close_vs_mean_50",
    "close_vs_mean_150",
    "close_vs_mean_200",
    "mean_200_trend_20",
    "above_low_252",
    "from_high_252",
    "up_down_volume_50",
    "rsi_14",
    "atr_ratio_14",
    "macd_signal_diff",
    "bollinger_width_20",
    "vol_ratio_63_21",
]


def validate_rebuild_config(cfg: Dict[str, Any]) -> None:
    """Strictly validate the rebuild configuration dictionary against the specification contract."""
    # 1. NaN and Inf check
    validate_no_nan(cfg)

    # 2. Key check at root level
    _check_allowed_keys(cfg, ROOT_ALLOWED_KEYS, "root configuration")

    # Required top-level fields
    mandatory_root = [
        "schema_version", "created_date", "status", "production_authorized",
        "authoritative_document", "scope", "data", "folds", "representation",
        "neural", "ridge", "memory", "integration", "arms", "execution",
        "analysis", "cost_stress", "counts", "operations", "production_gates"
    ]
    for m in mandatory_root:
        if m not in cfg:
            raise ContractValidationError(f"Missing mandatory root configuration field: {m}")

    if cfg["schema_version"] != "rebuild-proposal-1":
        raise ContractValidationError(f"Unexpected schema_version: {cfg['schema_version']}")

    # 3. Scope validation
    scope = cfg["scope"]
    _check_allowed_keys(scope, SCOPE_ALLOWED_KEYS, "scope")
    if scope.get("namespace") != "memory_study_v2":
        raise ContractValidationError(f"Invalid namespace: {scope.get('namespace')}")
    if scope.get("untouched_holdout_claim") is not False:
        raise ContractValidationError("untouched_holdout_claim must be false (exposure audited)")
    if scope.get("evaluation_status") != "previously_exposed_historical_walk_forward":
        raise ContractValidationError(
            f"Invalid evaluation_status: {scope.get('evaluation_status')}"
        )

    # 4. Data validation
    data = cfg["data"]
    _check_allowed_keys(data, DATA_ALLOWED_KEYS, "data")
    if data.get("primary_count") != 103:
        raise ContractValidationError(f"primary_count must be 103, got {data.get('primary_count')}")
    if data.get("requested_count") != 108:
        raise ContractValidationError(f"requested_count must be 108, got {data.get('requested_count')}")
    if data.get("point_in_time_universe_claim") is not False:
        raise ContractValidationError("point_in_time_universe_claim must be false")
    if data.get("impute_unknown_labels") is not False:
        raise ContractValidationError("impute_unknown_labels must be false")
    if data.get("backfill_prices") is not False:
        raise ContractValidationError("backfill_prices must be false")
    _check_allowed_keys(data["settings"], DATA_SETTINGS_ALLOWED_KEYS, "data.settings")

    # 5. Folds validation
    folds = cfg["folds"]
    if not isinstance(folds, list) or len(folds) != 6:
        raise ContractValidationError(f"Exactly 6 folds required, got {len(folds) if isinstance(folds, list) else type(folds)}")
    
    expected_years = [2020, 2021, 2022, 2023, 2024, 2025]
    for idx, f in enumerate(folds):
        _check_allowed_keys(f, FOLD_ALLOWED_KEYS, f"folds[{idx}]")
        ey = expected_years[idx]
        if f["evaluation_year"] != ey:
            raise ContractValidationError(f"Fold {idx} evaluation_year expected {ey}, got {f['evaluation_year']}")
        # Check chronology
        if not (f["train_origin_start"] <= f["training_availability_cutoff"] <
                f["validation_query_start"] <= f["validation_query_end"] <= f["validation_label_cutoff"] <
                f["development_query_start"] <= f["development_query_end"] <= f["development_label_cutoff"] <
                f["evaluation_query_start"] <= f["evaluation_query_end"]):
            raise ContractValidationError(f"Fold {ey} violates chronological ordering or disjoint windows")
        if f["bank_cutoff"] != f["training_availability_cutoff"]:
            raise ContractValidationError(f"Fold {ey} bank_cutoff must match training_availability_cutoff")

    # 6. Representation validation
    rep = cfg["representation"]
    _check_allowed_keys(rep, REPRESENTATION_ALLOWED_KEYS, "representation")
    if rep.get("features") != 23:
        raise ContractValidationError(f"representation.features must be 23, got {rep.get('features')}")
    if rep.get("window_sessions") != 252:
        raise ContractValidationError(f"window_sessions must be 252, got {rep.get('window_sessions')}")
    if rep.get("patch_sessions") != 6:
        raise ContractValidationError(f"patch_sessions must be 6, got {rep.get('patch_sessions')}")
    if rep.get("patch_count") != 42:
        raise ContractValidationError(f"patch_count must be 42, got {rep.get('patch_count')}")
    if rep.get("flattened_dim") != 966:
        raise ContractValidationError(f"flattened_dim must be 966, got {rep.get('flattened_dim')}")
    if rep.get("exclude_decision_session") is not True:
        raise ContractValidationError("exclude_decision_session must be true")
    if rep.get("feature_standard_deviation_ddof") != 1:
        raise ContractValidationError("feature_standard_deviation_ddof must be 1")
    if rep.get("scaler_ddof") != 0:
        raise ContractValidationError("scaler_ddof must be 0")
    if rep.get("clip") != [-5, 5]:
        raise ContractValidationError(f"clip must be [-5, 5], got {rep.get('clip')}")
    if rep.get("target_sessions") != 63:
        raise ContractValidationError(f"target_sessions must be 63, got {rep.get('target_sessions')}")

    # 7. Neural validation
    neural = cfg["neural"]
    _check_allowed_keys(neural, NEURAL_ALLOWED_KEYS, "neural")
    if neural.get("seeds") != [7, 17, 37]:
        raise ContractValidationError(f"neural.seeds must be [7, 17, 37], got {neural.get('seeds')}")
    if neural.get("fresh_initialization") is not True:
        raise ContractValidationError("fresh_initialization must be true")
    if neural.get("legacy_checkpoint_loading") is not False:
        raise ContractValidationError("legacy_checkpoint_loading must be false")
    if neural.get("early_stop_patience") != 5:
        raise ContractValidationError(f"early_stop_patience must be 5, got {neural.get('early_stop_patience')}")
    if neural.get("min_epochs") != 5 or neural.get("max_epochs") != 50:
        raise ContractValidationError("epochs must be min 5, max 50")
    if neural.get("deterministic_algorithms") is not True:
        raise ContractValidationError("deterministic_algorithms must be true")

    # 8. Ridge validation
    ridge = cfg["ridge"]
    _check_allowed_keys(ridge, RIDGE_ALLOWED_KEYS, "ridge")
    if ridge.get("l2_coefficient_on_equal_market_mse") != 0.001:
        raise ContractValidationError("ridge L2 coefficient must be 0.001")
    if ridge.get("penalize_intercept") is not False:
        raise ContractValidationError("penalize_intercept must be false")
    if ridge.get("dtype") != "float64":
        raise ContractValidationError("ridge dtype must be float64")

    # 9. Memory validation
    memory = cfg["memory"]
    _check_allowed_keys(memory, MEMORY_ALLOWED_KEYS, "memory")
    if memory.get("k") != 25:
        raise ContractValidationError(f"memory.k must be 25, got {memory.get('k')}")
    if memory.get("max_records_per_security") != 3:
        raise ContractValidationError("max_records_per_security must be 3")
    if memory.get("minimum_origin_spacing_sessions") != 21:
        raise ContractValidationError("minimum_origin_spacing_sessions must be 21")
    if memory.get("same_canonical_security_excluded") is not True:
        raise ContractValidationError("same_canonical_security_excluded must be true")
    if memory.get("random_master_seeds") != [1001, 1002, 1003]:
        raise ContractValidationError(f"random_master_seeds must be [1001, 1002, 1003], got {memory.get('random_master_seeds')}")
    if memory.get("maturity_sessions") != 126:
        raise ContractValidationError(f"maturity_sessions must be 126, got {memory.get('maturity_sessions')}")

    # 10. Integration validation
    integ = cfg["integration"]
    _check_allowed_keys(integ, INTEGRATION_ALLOWED_KEYS, "integration")
    expected_gate_diag = ["abs(base)", "abs(base-memory)", "abs(memory)"]
    if integ.get("gate_diagnostics") != expected_gate_diag:
        raise ContractValidationError(f"gate_diagnostics must be {expected_gate_diag}")
    if integ.get("gate_steps") != 50:
        raise ContractValidationError("gate_steps must be 50")
    if integ.get("mixture_grid") != [0, 0.1, 0.25, 0.5, 1.0]:
        raise ContractValidationError(f"mixture_grid must be [0, 0.1, 0.25, 0.5, 1.0], got {integ.get('mixture_grid')}")

    # 11. Arms validation
    arms = cfg["arms"]
    if not isinstance(arms, list) or len(arms) != 16:
        raise ContractValidationError(f"arms must contain exactly 16 configurations, got {len(arms)}")
    actual_arms = []
    total_realizations = 0
    for idx, arm in enumerate(arms):
        _check_allowed_keys(arm, ARM_ALLOWED_KEYS, f"arms[{idx}]")
        actual_arms.append((arm["id"], arm["realizations_per_market"]))
        total_realizations += arm["realizations_per_market"]
    if actual_arms != EXPECTED_16_ARMS:
        raise ContractValidationError(
            f"Arms mismatch. Expected: {EXPECTED_16_ARMS} Actual: {actual_arms}"
        )
    if total_realizations != 34:
        raise ContractValidationError(f"Total realizations per market must be 34, got {total_realizations}")

    # 12. Execution validation
    exec_cfg = cfg["execution"]
    _check_allowed_keys(exec_cfg, EXECUTION_ALLOWED_KEYS, "execution")
    if exec_cfg.get("initial_capital_account_units") != 100000:
        raise ContractValidationError("initial_capital_account_units must be 100000")
    if exec_cfg.get("max_positions") != 3:
        raise ContractValidationError("max_positions must be 3")
    if exec_cfg.get("cash_budget_fraction") != 0.95:
        raise ContractValidationError("cash_budget_fraction must be 0.95")
    if exec_cfg.get("commission_per_side") != 0.001:
        raise ContractValidationError("commission_per_side must be 0.001")
    if exec_cfg.get("slippage_per_side") != 0.0005:
        raise ContractValidationError("slippage_per_side must be 0.0005")
    if exec_cfg.get("fill") != "next_scheduled_open":
        raise ContractValidationError("fill must be next_scheduled_open")
    if exec_cfg.get("exit_before_entry") is not True:
        raise ContractValidationError("exit_before_entry must be true")
    if exec_cfg.get("maximum_holding_sessions") != 63:
        raise ContractValidationError("maximum_holding_sessions must be 63")
    if exec_cfg.get("minimum_stop_fraction") != 0.1:
        raise ContractValidationError("minimum_stop_fraction must be 0.1")
    if exec_cfg.get("atr_stop_multiplier") != 2.5:
        raise ContractValidationError("atr_stop_multiplier must be 2.5")

    # 13. Analysis validation
    analysis = cfg["analysis"]
    _check_allowed_keys(analysis, ANALYSIS_ALLOWED_KEYS, "analysis")
    if analysis.get("annualization") != 252:
        raise ContractValidationError("annualization must be 252")
    if analysis.get("bootstrap_draws") != 10000:
        raise ContractValidationError(f"bootstrap_draws must be 10000, got {analysis.get('bootstrap_draws')}")
    if analysis.get("bootstrap_primary_weeks") != 4:
        raise ContractValidationError("bootstrap_primary_weeks must be 4")
    if analysis.get("bootstrap_sensitivity_weeks") != [2, 8]:
        raise ContractValidationError("bootstrap_sensitivity_weeks must be [2, 8]")
    contrasts = analysis.get("primary_contrasts", [])
    if len(contrasts) != 8:
        raise ContractValidationError(f"primary_contrasts must have 8 pairs (P1..P8), got {len(contrasts)}")
    for idx, c in enumerate(contrasts):
        _check_allowed_keys(c, CONTRAST_ALLOWED_KEYS, f"analysis.primary_contrasts[{idx}]")

    # 14. Cost stress validation
    cost = cfg["cost_stress"]
    _check_allowed_keys(cost, COST_STRESS_ALLOWED_KEYS, "cost_stress")
    expected_cost_arms = ["MEM_SIM", "HIST_PRIOR", "MLP_GATE", "TRANS_GATE"]
    if cost.get("arms") != expected_cost_arms:
        raise ContractValidationError(f"cost_stress.arms must be {expected_cost_arms}")
    if cost.get("slippage_per_side") != [0, 0.0015]:
        raise ContractValidationError("cost_stress.slippage_per_side must be [0, 0.0015]")
    if cost.get("commission_per_side") != 0.001:
        raise ContractValidationError("cost_stress.commission_per_side must be 0.001")

    # 15. Counts validation
    counts = cfg["counts"]
    _check_allowed_keys(counts, COUNTS_ALLOWED_KEYS, "counts")
    expected_counts = {
        "folds": 6,
        "markets": 6,
        "neural_selected_fits": 36,
        "gate_fits": 36,
        "ridge_fits": 6,
        "configurations": 16,
        "logical_realizations_per_market": 34,
        "primary_continuous_paths": 204,
        "primary_logical_market_year_cells": 1224,
        "cost_stress_additional_continuous_paths": 96,
    }
    for k, ev in expected_counts.items():
        if counts.get(k) != ev:
            raise ContractValidationError(f"counts.{k} expected {ev}, got {counts.get(k)}")

    # 16. Operations validation
    ops = cfg["operations"]
    _check_allowed_keys(ops, OPERATIONS_ALLOWED_KEYS, "operations")
    _check_allowed_keys(ops["laptop"], OPERATIONS_PROFILE_ALLOWED_KEYS, "operations.laptop")
    _check_allowed_keys(ops["a30"], OPERATIONS_PROFILE_ALLOWED_KEYS, "operations.a30")

    # 17. Production gates validation
    gates = cfg["production_gates"]
    _check_allowed_keys(gates, PRODUCTION_GATES_ALLOWED_KEYS, "production_gates")

    is_prod = cfg.get("production_authorized", False)
    if is_prod:
        unresolved = [k for k, v in gates.items() if v is None or v is False]
        if unresolved:
            raise ContractValidationError(
                f"Production execution unauthorized: unresolved production gates: {unresolved}"
            )
        if cfg.get("status") != "production_authorized":
            raise ContractValidationError(
                f"Status must be 'production_authorized' when production_authorized is true, got '{cfg.get('status')}'"
            )
    else:
        if cfg.get("status") != "implementation_and_pilot_only":
            raise ContractValidationError(
                f"Status must be 'implementation_and_pilot_only' when production_authorized is false, got '{cfg.get('status')}'"
            )


def validate_universe_request(csv_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Validate universe_request.csv against contract: 108 requests, 103 primary, 6 markets, preserved leading zeros."""
    path = Path(csv_path)
    if not path.exists():
        raise ContractValidationError(f"Universe request file does not exist: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = [
            "project_security_id", "market", "vendor_symbol", "account_currency",
            "in_primary", "legacy_exclusion_reason", "exchange_mic", "timezone",
            "quote_unit", "quote_to_account_multiplier", "issuer_id", "metadata_verified"
        ]
        if reader.fieldnames != required_cols:
            raise ContractValidationError(
                f"universe_request.csv headers mismatch. Expected: {required_cols} Actual: {reader.fieldnames}"
            )
        for r in reader:
            rows.append(r)

    if len(rows) != 108:
        raise ContractValidationError(f"universe_request.csv must have exactly 108 rows, got {len(rows)}")

    ids = [r["project_security_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ContractValidationError("Duplicate project_security_id found in universe_request.csv")

    primary_rows = [r for r in rows if r["in_primary"].lower() in ("true", "1")]
    excluded_rows = [r for r in rows if r["in_primary"].lower() in ("false", "0")]

    if len(primary_rows) != 103:
        raise ContractValidationError(f"Expected 103 primary securities, got {len(primary_rows)}")
    if len(excluded_rows) != 5:
        raise ContractValidationError(f"Expected 5 excluded securities, got {len(excluded_rows)}")

    # Check 6 markets distribution in primary
    market_counts: Dict[str, int] = {}
    for r in primary_rows:
        m = r["market"]
        market_counts[m] = market_counts.get(m, 0) + 1

    expected_market_counts = {
        "Brazil": 15,
        "China": 18,
        "France": 17,
        "India": 18,
        "UK": 17,
        "US": 18,
    }
    if market_counts != expected_market_counts:
        raise ContractValidationError(
            f"Primary market counts mismatch. Expected: {expected_market_counts} Actual: {market_counts}"
        )

    # Check that leading zeros are preserved for China tickers (e.g. 000001.SZ)
    china_tickers = [r["vendor_symbol"] for r in primary_rows if r["market"] == "China"]
    leading_zero_tickers = [t for t in china_tickers if t.startswith("0")]
    if not leading_zero_tickers:
        raise ContractValidationError("No China ticker with leading zeros found - possible zero-stripping corruption")

    return rows


def validate_feature_contract(csv_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Validate feature_contract.csv: 23 features in exact order, unique names."""
    path = Path(csv_path)
    if not path.exists():
        raise ContractValidationError(f"Feature contract file does not exist: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = [
            "index", "name", "legacy_alias", "formula",
            "minimum_bar_count", "input_basis", "missing_policy"
        ]
        if reader.fieldnames != required_cols:
            raise ContractValidationError(
                f"feature_contract.csv headers mismatch. Expected: {required_cols} Actual: {reader.fieldnames}"
            )
        for r in reader:
            rows.append(r)

    if len(rows) != 23:
        raise ContractValidationError(f"feature_contract.csv must have exactly 23 rows, got {len(rows)}")

    actual_names = [r["name"] for r in rows]
    if actual_names != EXPECTED_FEATURES_ORDERED:
        raise ContractValidationError(
            f"Feature names/order mismatch. Expected: {EXPECTED_FEATURES_ORDERED} Actual: {actual_names}"
        )

    for idx, r in enumerate(rows):
        if int(r["index"]) != idx:
            raise ContractValidationError(f"Row index mismatch: expected {idx}, got {r['index']}")

    return rows


def load_and_validate_rebuild_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON from config_path and validate strictly against the rebuild contract."""
    p = Path(config_path)
    if not p.exists():
        raise ContractValidationError(f"Configuration file does not exist: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    validate_rebuild_config(data)
    return data
