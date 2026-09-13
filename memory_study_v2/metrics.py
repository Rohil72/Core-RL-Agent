"""Portfolio performance and forecast accuracy metrics (v2).

Acceptance criteria addressed:
- A28: Annualization, zero variance, initial/terminal costs, turnover and exposure match manual fixtures.
- A29: Constant prior rank IC is missing; valid-mask and daily cross-sectional averaging tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class PortfolioSummaryMetrics:
    annualized_return: float
    annualized_volatility: float
    annualized_sharpe: float
    max_drawdown: float
    win_rate: float
    annualized_turnover: float
    mean_exposure: float
    num_sessions: int


def compute_portfolio_metrics(
    daily_navs: np.ndarray,
    executed_notionals: np.ndarray,
    holdings_values: np.ndarray,
    initial_capital: float = 100000.0,
    annualization: int = 252,
) -> PortfolioSummaryMetrics:
    """Compute portfolio statistics according to Section 9.1 contract."""
    N = len(daily_navs)
    if N == 0:
        return PortfolioSummaryMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    # Prepend initial capital to compute returns including initial entry costs
    nav_series = np.concatenate(([initial_capital], daily_navs))
    returns = nav_series[1:] / nav_series[:-1] - 1.0

    # Fail on bankrupt/negative NAV
    if np.any(daily_navs <= 0) or np.any(returns <= -1.0):
        raise ValueError("Bankrupt or non-positive equity detected in unlevered portfolio run")

    # Annualized return = exp(252 * mean(log(1 + r))) - 1
    mean_log1p = float(np.mean(np.log1p(returns)))
    ann_return = float(math.exp(annualization * mean_log1p) - 1.0)

    # Annualized volatility = sqrt(252) * population_std(r) (ddof=0)
    pop_std = float(np.std(returns, ddof=0))
    ann_vol = float(math.sqrt(annualization) * pop_std)

    # Annualized Sharpe = sqrt(252) * mean(r) / pop_std (0 if pop_std <= 1e-8)
    mean_r = float(np.mean(returns))
    if pop_std <= 1e-8:
        ann_sharpe = 0.0
    else:
        ann_sharpe = float(math.sqrt(annualization) * mean_r / pop_std)

    # Max drawdown including initial capital in running peak
    peaks = np.maximum.accumulate(nav_series)
    drawdowns = (nav_series - peaks) / peaks
    max_dd = float(np.min(drawdowns))

    # Win rate: fraction of sessions r > 0
    win_rate = float(np.mean(returns > 0.0))

    # Annualized turnover = (252 / N) * sum(abs(notional)) / (2 * mean(E))
    mean_equity = float(np.mean(daily_navs))
    total_notional = float(np.sum(np.abs(executed_notionals)))
    ann_turnover = float((annualization / float(N)) * total_notional / (2.0 * mean_equity)) if mean_equity > 0 else 0.0

    # Exposure = mean(holdings / E)
    mean_exposure = float(np.mean(holdings_values / daily_navs)) if mean_equity > 0 else 0.0

    return PortfolioSummaryMetrics(
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        annualized_sharpe=ann_sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        annualized_turnover=ann_turnover,
        mean_exposure=mean_exposure,
        num_sessions=N,
    )


def compute_daily_cross_sectional_rank_ic(
    dates: List[str],
    predictions: np.ndarray,
    targets: np.ndarray,
    min_assets: int = 5,
) -> Tuple[Optional[float], int]:
    """Compute daily within-market Spearman rank IC (Section 9.1, A29).

    Requires at least 5 assets and non-constant forecasts/outcomes.
    Undefined IC is missing/NaN (NEVER 0.0). Returns (mean_defined_ic, defined_date_count).
    """
    df = pd.DataFrame({
        "date": dates,
        "pred": predictions,
        "target": targets,
    })

    daily_ics: List[float] = []
    for d, group in df.groupby("date"):
        if len(group) < min_assets:
            continue
        p = group["pred"].to_numpy()
        y = group["target"].to_numpy()

        # Check for constant values
        if np.all(p == p[0]) or np.all(y == y[0]):
            # Constant prior or degenerate target: rank IC is undefined (missing, not 0)
            continue

        res = spearmanr(p, y)
        ic = float(res.statistic)
        if not math.isnan(ic):
            daily_ics.append(ic)

    if not daily_ics:
        return None, 0
    return float(np.mean(daily_ics)), len(daily_ics)
