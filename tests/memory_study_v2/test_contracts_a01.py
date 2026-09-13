"""Acceptance Test A01: Schema and configuration contract validation."""

import copy
import json
import pytest
from pathlib import Path

from memory_study_v2.contracts import (
    ContractValidationError,
    load_and_validate_rebuild_config,
    validate_rebuild_config,
    validate_universe_request,
    validate_feature_contract,
    to_canonical_json,
    validate_no_nan,
)

CONFIG_PATH = Path("rebuild_plan/config.proposed.json")
UNIVERSE_PATH = Path("rebuild_plan/universe_request.csv")
FEATURE_PATH = Path("rebuild_plan/feature_contract.csv")


def test_valid_proposed_config_passes():
    """Verify that the official rebuild_plan/config.proposed.json passes validation."""
    cfg = load_and_validate_rebuild_config(CONFIG_PATH)
    assert cfg["schema_version"] == "rebuild-proposal-1"
    assert cfg["production_authorized"] is False
    assert cfg["status"] == "implementation_and_pilot_only"
    assert cfg["counts"]["primary_continuous_paths"] == 204
    assert cfg["counts"]["primary_logical_market_year_cells"] == 1224


def test_unknown_root_key_rejects():
    """Verify that an unknown root key in configuration is strictly rejected."""
    cfg = load_and_validate_rebuild_config(CONFIG_PATH)
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["rogue_parameter"] = 42
    with pytest.raises(ContractValidationError, match="Unknown configuration keys in root"):
        validate_rebuild_config(bad_cfg)


def test_unknown_nested_key_rejects():
    """Verify that an unknown nested key in configuration is strictly rejected."""
    cfg = load_and_validate_rebuild_config(CONFIG_PATH)
    
    # Nested in neural
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["neural"]["unauthorized_optimizer"] = "RMSprop"
    with pytest.raises(ContractValidationError, match="Unknown configuration keys in neural"):
        validate_rebuild_config(bad_cfg)

    # Nested in execution
    bad_cfg2 = copy.deepcopy(cfg)
    bad_cfg2["execution"]["hidden_multiplier"] = 1.5
    with pytest.raises(ContractValidationError, match="Unknown configuration keys in execution"):
        validate_rebuild_config(bad_cfg2)


def test_nan_in_config_rejects():
    """Verify that NaN float values in configuration are strictly rejected."""
    cfg = load_and_validate_rebuild_config(CONFIG_PATH)
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["execution"]["slippage_per_side"] = float("nan")
    with pytest.raises(ContractValidationError, match="NaN value detected"):
        validate_rebuild_config(bad_cfg)


def test_inf_in_config_rejects():
    """Verify that Infinite float values in configuration are strictly rejected."""
    cfg = load_and_validate_rebuild_config(CONFIG_PATH)
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["neural"]["learning_rate"] = float("inf")
    with pytest.raises(ContractValidationError, match="Infinite value detected"):
        validate_rebuild_config(bad_cfg)


def test_missing_mandatory_field_rejects():
    """Verify that omitting a mandatory root field is rejected."""
    cfg = load_and_validate_rebuild_config(CONFIG_PATH)
    bad_cfg = copy.deepcopy(cfg)
    del bad_cfg["arms"]
    with pytest.raises(ContractValidationError, match="Missing mandatory root configuration field: arms"):
        validate_rebuild_config(bad_cfg)


def test_production_authorized_without_resolved_gates_rejects():
    """Verify that claiming production authorization with unresolved gates rejects."""
    cfg = load_and_validate_rebuild_config(CONFIG_PATH)
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["production_authorized"] = True
    bad_cfg["status"] = "production_authorized"
    # production gates remain null
    with pytest.raises(ContractValidationError, match="Production execution unauthorized: unresolved production gates"):
        validate_rebuild_config(bad_cfg)


def test_universe_request_validation():
    """Verify universe_request.csv: 108 requests, 103 primary, 6 markets, preserved leading zeros."""
    rows = validate_universe_request(UNIVERSE_PATH)
    assert len(rows) == 108
    primary = [r for r in rows if r["in_primary"].lower() in ("true", "1")]
    assert len(primary) == 103
    # Check leading zero preservation (China ticker 000001.SZ)
    china_symbols = [r["vendor_symbol"] for r in primary if r["market"] == "China"]
    assert "000333.SZ" in china_symbols
    assert any(s.startswith("00") for s in china_symbols)


def test_feature_contract_validation():
    """Verify feature_contract.csv: 23 features in exact order."""
    rows = validate_feature_contract(FEATURE_PATH)
    assert len(rows) == 23
    assert rows[0]["name"] == "return_1"
    assert rows[9]["name"] == "drawdown_252"
    assert rows[22]["name"] == "vol_ratio_63_21"


def test_canonical_json_determinism():
    """Verify that canonical JSON serializes with sorted keys and compact separators."""
    d = {"b": 2, "a": 1, "nested": {"z": 10, "y": 20}}
    canon = to_canonical_json(d)
    assert canon == '{"a":1,"b":2,"nested":{"y":20,"z":10}}'
