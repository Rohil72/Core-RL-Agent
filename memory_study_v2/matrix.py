"""Configuration matrix scope, path inventory, and cell reconciliation (v2).

Acceptance criteria addressed:
- A36: 16 arms, 34 realizations, 204 primary paths, 1224 cells and 96 stress paths reconcile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

# 16 Configurations defined in Section 7 of REBUILD_IMPLEMENTATION_PLAN.md:
PRIMARY_CONFIGURATIONS = [
    "MEM_SIM",
    "KNN_PLAIN",
    "MEM_RANDOM",
    "HIST_PRIOR",
    "RIDGE_ANNUAL",
    "MLP_BASE",
    "TRANS_BASE",
    "MLP_MIX_MSE",
    "TRANS_MIX_MSE",
    "MLP_MIX_SR",
    "TRANS_MIX_SR",
    "MLP_GATE",
    "TRANS_GATE",
    "MOMENTUM_21",
    "VOL_MOMENTUM_21",
    "PASSIVE_EQUAL_WEIGHT",
]

# Realization seeds for neural and random models
NEURAL_SEEDS = [7, 17, 37]
RANDOM_SEEDS = [7, 17, 37]

# 6 Evaluation Markets
MARKETS = ["US", "IN", "CN", "FR", "GB", "BR"]

# 6 Annual Walk-Forward Evaluation Folds
EVALUATION_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# 4 Cost-Stress Configurations
COST_STRESS_CONFIGS = ["MEM_SIM", "HIST_PRIOR", "MLP_GATE", "TRANS_GATE"]

# Cost stress slippage settings (commission fixed at 0.001)
COST_STRESS_SLIPPAGES = [0.0, 0.0015]


@dataclass(frozen=True)
class RealizationKey:
    configuration: str
    seed: Optional[int]
    is_deterministic: bool

    @property
    def realization_id(self) -> str:
        if self.seed is not None:
            return f"{self.configuration}_seed{self.seed}"
        return self.configuration


@dataclass(frozen=True)
class ContinuousPathKey:
    market: str
    configuration: str
    seed: Optional[int]

    @property
    def path_id(self) -> str:
        if self.seed is not None:
            return f"{self.market}_{self.configuration}_seed{self.seed}"
        return f"{self.market}_{self.configuration}"


@dataclass(frozen=True)
class CellKey:
    year: int
    market: str
    configuration: str
    seed: Optional[int]

    @property
    def cell_id(self) -> str:
        if self.seed is not None:
            return f"{self.year}_{self.market}_{self.configuration}_seed{self.seed}"
        return f"{self.year}_{self.market}_{self.configuration}"


@dataclass(frozen=True)
class CostStressPathKey:
    market: str
    configuration: str
    seed: Optional[int]
    slippage: float

    @property
    def stress_id(self) -> str:
        s_str = str(self.slippage).replace(".", "p")
        if self.seed is not None:
            return f"{self.market}_{self.configuration}_seed{self.seed}_slip{s_str}"
        return f"{self.market}_{self.configuration}_slip{s_str}"


def get_realizations_per_market() -> List[RealizationKey]:
    """Generate the 34 logical realizations per market."""
    realizations: List[RealizationKey] = []
    
    # 7 Deterministic realizations (1 each)
    deterministic_configs = [
        "MEM_SIM",
        "KNN_PLAIN",
        "HIST_PRIOR",
        "RIDGE_ANNUAL",
        "MOMENTUM_21",
        "VOL_MOMENTUM_21",
        "PASSIVE_EQUAL_WEIGHT",
    ]
    for c in deterministic_configs:
        realizations.append(RealizationKey(configuration=c, seed=None, is_deterministic=True))

    # 3 Random-memory realizations
    for s in RANDOM_SEEDS:
        realizations.append(RealizationKey(configuration="MEM_RANDOM", seed=s, is_deterministic=False))

    # 24 Neural-derived realizations (8 configs * 3 seeds)
    neural_configs = [
        "MLP_BASE",
        "TRANS_BASE",
        "MLP_MIX_MSE",
        "TRANS_MIX_MSE",
        "MLP_MIX_SR",
        "TRANS_MIX_SR",
        "MLP_GATE",
        "TRANS_GATE",
    ]
    for c in neural_configs:
        for s in NEURAL_SEEDS:
            realizations.append(RealizationKey(configuration=c, seed=s, is_deterministic=False))

    return realizations


def generate_primary_paths() -> List[ContinuousPathKey]:
    """Generate the 204 continuous market-policy paths across 6 markets."""
    realizations = get_realizations_per_market()
    paths: List[ContinuousPathKey] = []
    for m in MARKETS:
        for r in realizations:
            paths.append(ContinuousPathKey(market=m, configuration=r.configuration, seed=r.seed))
    return paths


def generate_logical_cells() -> List[CellKey]:
    """Generate the 1,224 logical market-year cells across 6 evaluation years."""
    paths = generate_primary_paths()
    cells: List[CellKey] = []
    for y in EVALUATION_YEARS:
        for p in paths:
            cells.append(CellKey(year=y, market=p.market, configuration=p.configuration, seed=p.seed))
    return cells


def generate_cost_stress_paths() -> List[CostStressPathKey]:
    """Generate the 96 mandatory cost stress paths."""
    realizations = get_realizations_per_market()
    stress_realizations = [r for r in realizations if r.configuration in COST_STRESS_CONFIGS]
    
    # 2 stress settings * 6 markets * 8 realizations = 96 paths
    stress_paths: List[CostStressPathKey] = []
    for slip in COST_STRESS_SLIPPAGES:
        for m in MARKETS:
            for r in stress_realizations:
                stress_paths.append(
                    CostStressPathKey(
                        market=m,
                        configuration=r.configuration,
                        seed=r.seed,
                        slippage=slip,
                    )
                )
    return stress_paths


def reconcile_configuration_matrix() -> Dict[str, Any]:
    """Reconcile and verify the full configuration matrix scope."""
    realizations = get_realizations_per_market()
    paths = generate_primary_paths()
    cells = generate_logical_cells()
    stress_paths = generate_cost_stress_paths()

    res = {
        "num_configurations": len(PRIMARY_CONFIGURATIONS),
        "num_realizations_per_market": len(realizations),
        "num_deterministic_realizations": sum(1 for r in realizations if r.is_deterministic),
        "num_random_realizations": sum(1 for r in realizations if r.configuration == "MEM_RANDOM"),
        "num_neural_realizations": sum(
            1 for r in realizations if r.configuration not in ["MEM_RANDOM"] and not r.is_deterministic
        ),
        "num_markets": len(MARKETS),
        "num_primary_continuous_paths": len(paths),
        "num_evaluation_years": len(EVALUATION_YEARS),
        "num_logical_market_year_cells": len(cells),
        "num_cost_stress_slippage_settings": len(COST_STRESS_SLIPPAGES),
        "num_cost_stress_configs": len(COST_STRESS_CONFIGS),
        "num_cost_stress_paths": len(stress_paths),
        "reconciled": (
            len(PRIMARY_CONFIGURATIONS) == 16
            and len(realizations) == 34
            and len(paths) == 204
            and len(cells) == 1224
            and len(stress_paths) == 96
        ),
    }
    return res
