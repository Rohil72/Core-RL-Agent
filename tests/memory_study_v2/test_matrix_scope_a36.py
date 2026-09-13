"""Acceptance Test A36: Configuration matrix scope reconciliation.

Acceptance criteria:
- 16 arms, 34 realizations, 204 primary paths, 1224 cells and 96 stress paths reconcile.
"""

import pytest

from memory_study_v2.matrix import (
    COST_STRESS_CONFIGS,
    COST_STRESS_SLIPPAGES,
    EVALUATION_YEARS,
    MARKETS,
    PRIMARY_CONFIGURATIONS,
    generate_cost_stress_paths,
    generate_logical_cells,
    generate_primary_paths,
    get_realizations_per_market,
    reconcile_configuration_matrix,
)


def test_matrix_reconciliation():
    rec = reconcile_configuration_matrix()
    assert rec["num_configurations"] == 16
    assert rec["num_realizations_per_market"] == 34
    assert rec["num_deterministic_realizations"] == 7
    assert rec["num_random_realizations"] == 3
    assert rec["num_neural_realizations"] == 24
    assert rec["num_markets"] == 6
    assert rec["num_primary_continuous_paths"] == 204
    assert rec["num_evaluation_years"] == 6
    assert rec["num_logical_market_year_cells"] == 1224
    assert rec["num_cost_stress_paths"] == 96
    assert rec["reconciled"] is True


def test_matrix_unique_keys():
    # 1. Check Realizations uniqueness
    realizations = get_realizations_per_market()
    assert len(realizations) == 34
    realization_ids = [r.realization_id for r in realizations]
    assert len(set(realization_ids)) == 34, "Duplicate realization IDs found"

    # 2. Check Primary Paths uniqueness
    paths = generate_primary_paths()
    assert len(paths) == 204
    path_ids = [p.path_id for p in paths]
    assert len(set(path_ids)) == 204, "Duplicate continuous path IDs found"

    # 3. Check Cells uniqueness
    cells = generate_logical_cells()
    assert len(cells) == 1224
    cell_ids = [c.cell_id for c in cells]
    assert len(set(cell_ids)) == 1224, "Duplicate logical cell IDs found"

    # 4. Check Cost Stress Paths uniqueness
    stress_paths = generate_cost_stress_paths()
    assert len(stress_paths) == 96
    stress_ids = [s.stress_id for s in stress_paths]
    assert len(set(stress_ids)) == 96, "Duplicate cost stress path IDs found"


def test_cost_stress_arm_membership():
    stress_paths = generate_cost_stress_paths()
    stress_configs = {p.configuration for p in stress_paths}
    assert stress_configs == {"MEM_SIM", "HIST_PRIOR", "MLP_GATE", "TRANS_GATE"}
    stress_slippages = {p.slippage for p in stress_paths}
    assert stress_slippages == {0.0, 0.0015}
