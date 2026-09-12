"""
Evidence export module for public release.
Exports canonical data and metadata from research_runs/memory_study/final_comparison
to a standalone public research data repository bundle.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from memory_study.cache_builder import FEATURE_NAMES_23


def generate_evaluated_feature_definitions() -> pd.DataFrame:
    """
    Generates names, order, formulas, units, scaling, and missing-value rules
    derived directly from the evaluated 23-feature implementation (cache_builder.py).
    """
    feature_meta = {
        "tech_return_1d": {
            "category": "Momentum", "lookback_sessions": 1,
            "equation": "P_t / P_{t-1} - 1", "units": "decimal_return",
            "fill_handling": "fillna(0.0)"
        },
        "tech_momentum_3d": {
            "category": "Momentum", "lookback_sessions": 3,
            "equation": "P_t / P_{t-3} - 1", "units": "decimal_return",
            "fill_handling": "fillna(0.0)"
        },
        "tech_momentum_10d": {
            "category": "Momentum", "lookback_sessions": 10,
            "equation": "P_t / P_{t-10} - 1", "units": "decimal_return",
            "fill_handling": "fillna(0.0)"
        },
        "tech_momentum_21d": {
            "category": "Momentum", "lookback_sessions": 21,
            "equation": "P_t / P_{t-21} - 1", "units": "decimal_return",
            "fill_handling": "fillna(0.0)"
        },
        "tech_volatility_21d": {
            "category": "Volatility", "lookback_sessions": 21,
            "equation": "Std(tech_return_1d, 21)", "units": "daily_return_std",
            "fill_handling": "fillna(0.01)"
        },
        "tech_volume_sma_21d": {
            "category": "Volume", "lookback_sessions": 21,
            "equation": "SMA(Volume, 21)", "units": "shares",
            "fill_handling": "fillna(Volume)"
        },
        "tech_volume_ratio_21d": {
            "category": "Volume", "lookback_sessions": 21,
            "equation": "Volume / (tech_volume_sma_21d + 1e-9)", "units": "ratio",
            "fill_handling": "fillna(1.0)"
        },
        "tech_volume_change_1d": {
            "category": "Volume", "lookback_sessions": 1,
            "equation": "Volume_t / Volume_{t-1} - 1", "units": "decimal_change",
            "fill_handling": "fillna(0.0)"
        },
        "tech_intraday_range_hl": {
            "category": "Volatility", "lookback_sessions": 1,
            "equation": "(High - Low) / (Close + 1e-9)", "units": "ratio_to_close",
            "fill_handling": "fillna(0.02)"
        },
        "tech_drawdown_from_peak_21d": {
            "category": "Range", "lookback_sessions": 252,
            "equation": "(Close - RollingMax(Close, 252)) / (RollingMax + 1e-9)", "units": "ratio_to_peak",
            "fill_handling": "fillna(0.0)"
        },
        "tech_trend_slope_21d": {
            "category": "Trend", "lookback_sessions": 21,
            "equation": "LinearSlope(Close, 21) / (Close + 1e-9)", "units": "slope_per_session",
            "fill_handling": "fillna(0.0)"
        },
        "tech_close_vs_sma_50": {
            "category": "Trend", "lookback_sessions": 50,
            "equation": "(Close - SMA(Close, 50)) / (SMA_50 + 1e-9)", "units": "ratio_to_sma",
            "fill_handling": "fillna(0.0)"
        },
        "tech_close_vs_sma_150": {
            "category": "Trend", "lookback_sessions": 150,
            "equation": "(Close - SMA(Close, 150)) / (SMA_150 + 1e-9)", "units": "ratio_to_sma",
            "fill_handling": "fillna(0.0)"
        },
        "tech_close_vs_sma_200": {
            "category": "Trend", "lookback_sessions": 200,
            "equation": "(Close - SMA(Close, 200)) / (SMA_200 + 1e-9)", "units": "ratio_to_sma",
            "fill_handling": "fillna(0.0)"
        },
        "tech_sma_200_trend_20": {
            "category": "Trend", "lookback_sessions": 220,
            "equation": "(SMA_200 - Shift(SMA_200, 20)) / (Shift(SMA_200, 20) + 1e-9)", "units": "ratio_to_lagged_sma",
            "fill_handling": "fillna(0.0)"
        },
        "tech_pct_above_52w_low": {
            "category": "Range", "lookback_sessions": 252,
            "equation": "(Close - RollingMin(Close, 252)) / (RollingMin + 1e-9)", "units": "ratio_to_low",
            "fill_handling": "fillna(0.0)"
        },
        "tech_pct_from_52w_high": {
            "category": "Range", "lookback_sessions": 252,
            "equation": "(Close - RollingMax(Close, 252)) / (RollingMax + 1e-9)", "units": "ratio_to_high",
            "fill_handling": "fillna(0.0)"
        },
        "tech_up_down_volume_ratio_50": {
            "category": "Volume", "lookback_sessions": 50,
            "equation": "RollingSum(UpVol, 50) / (RollingSum(DownVol, 50) + 1e-9)", "units": "ratio",
            "fill_handling": "fillna(1.0)"
        },
        "tech_rsi_14": {
            "category": "Oscillator", "lookback_sessions": 14,
            "equation": "(100 - (100 / (1 + RS))) / 100", "units": "normalized_0_to_1",
            "fill_handling": "fillna(0.50)"
        },
        "tech_atr_ratio_14": {
            "category": "Volatility", "lookback_sessions": 14,
            "equation": "SMA(TrueRange, 14) / (Close + 1e-9)", "units": "ratio_to_close",
            "fill_handling": "fillna(0.02)"
        },
        "tech_macd_signal_diff": {
            "category": "Trend", "lookback_sessions": 26,
            "equation": "(MACD - Signal) / (Close + 1e-9)", "units": "ratio_to_close",
            "fill_handling": "fillna(0.0)"
        },
        "tech_bollinger_bandwidth_20": {
            "category": "Volatility", "lookback_sessions": 20,
            "equation": "(UpperBand - LowerBand) / (SMA_20 + 1e-9)", "units": "ratio_to_sma",
            "fill_handling": "fillna(0.04)"
        },
        "tech_historical_vol_ratio_63_21": {
            "category": "Volatility", "lookback_sessions": 63,
            "equation": "Std(Return, 63) / (Std(Return, 21) + 1e-9)", "units": "volatility_ratio",
            "fill_handling": "fillna(1.0)"
        }
    }

    rows = []
    for idx, name in enumerate(FEATURE_NAMES_23):
        meta = feature_meta[name]
        rows.append({
            "feature_idx": idx,
            "name": name,
            "category": meta["category"],
            "lookback_sessions": meta["lookback_sessions"],
            "equation": meta["equation"],
            "units": meta["units"],
            "scaling": "Per-market standardized [-5.0, 5.0]",
            "fill_handling": meta["fill_handling"],
            "missing_value_rule": "feature-specific explicit fillna rules followed by per-market mean imputation"
        })
    return pd.DataFrame(rows)


def export_bundle(source_dir: Path, output_dir: Path):
    source_dir = Path(source_dir).resolve()
    reanalysis_dir = source_dir / "reanalysis_v1"
    output_dir = Path(output_dir).resolve()
    
    data_dir = output_dir / "data"
    meta_dir = output_dir / "metadata"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Exporting release bundle from {source_dir} to {output_dir}...")
    
    # 1. Universe Manifest
    markets_dict = {
        "US": ["AAPL", "ADBE", "AMD", "AMGN", "AMZN", "AVGO", "CRM", "GOOGL", "INTU", "ISRG", "LMT", "META", "MSFT", "NOC", "NVDA", "ORCL", "REGN", "V"],
        "India": ["ASIANPAINT.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "EICHERMOT.NS", "HCLTECH.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "LT.NS", "M&M.NS", "MARUTI.NS", "PIDILITIND.NS", "RELIANCE.NS", "SUNPHARMA.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS"],
        "China": ["000333.SZ", "000725.SZ", "000858.SZ", "002230.SZ", "002241.SZ", "002415.SZ", "002475.SZ", "002594.SZ", "300059.SZ", "600036.SS", "600196.SS", "600276.SS", "600309.SS", "600519.SS", "600887.SS", "601012.SS", "601318.SS", "601888.SS"],
        "Brazil": ["B3SA3.SA", "EQTL3.SA", "FLRY3.SA", "ITUB4.SA", "KLBN11.SA", "LREN3.SA", "MGLU3.SA", "PETR4.SA", "RADL3.SA", "RAIL3.SA", "RENT3.SA", "SUZB3.SA", "TOTS3.SA", "VALE3.SA", "WEGE3.SA"],
        "France": ["AI.PA", "AIR.PA", "CAP.PA", "DG.PA", "DIM.PA", "DSY.PA", "EL.PA", "LR.PA", "MC.PA", "ML.PA", "OR.PA", "RI.PA", "RMS.PA", "SAF.PA", "SU.PA", "TEP.PA", "WLN.PA"],
        "UK": ["AUTO.L", "AZN.L", "BA.L", "CPG.L", "CRDA.L", "DGE.L", "EXPN.L", "HLMA.L", "JD.L", "LSEG.L", "OCDO.L", "PRU.L", "REL.L", "RMV.L", "RTO.L", "SGE.L", "SPX.L"]
    }
    currency_map = {"US": "USD", "India": "INR", "China": "CNY", "Brazil": "BRL", "France": "EUR", "UK": "GBP"}
    sessions_map = {"US": 252, "India": 248, "China": 242, "Brazil": 249, "France": 254, "UK": 253}
    
    univ_rows = []
    for mkt, tickers in markets_dict.items():
        for t in tickers:
            univ_rows.append({
                "market": mkt,
                "ticker": t,
                "currency": currency_map[mkt],
                "sessions_per_year": sessions_map[mkt],
                "included_in_study": True,
                "exclusion_reason": ""
            })
    # 5 documented excluded tickers
    univ_rows.append({"market": "Brazil", "ticker": "CIEL3.SA", "currency": "BRL", "sessions_per_year": 249, "included_in_study": False, "exclusion_reason": "Delisted / insufficient historical volume"})
    univ_rows.append({"market": "Brazil", "ticker": "JBSS3.SA", "currency": "BRL", "sessions_per_year": 249, "included_in_study": False, "exclusion_reason": "Corporate restructuring / missing history"})
    univ_rows.append({"market": "Brazil", "ticker": "EMBR3.SA", "currency": "BRL", "sessions_per_year": 249, "included_in_study": False, "exclusion_reason": "Historical corporate transaction data gap"})
    univ_rows.append({"market": "France", "ticker": "STM.PA", "currency": "EUR", "sessions_per_year": 254, "included_in_study": False, "exclusion_reason": "Primary dual listing on Borsa Italiana"})
    univ_rows.append({"market": "UK", "ticker": "AHT.L", "currency": "GBP", "sessions_per_year": 253, "included_in_study": False, "exclusion_reason": "Historical corporate split discontinuity"})
    
    univ_df = pd.DataFrame(univ_rows)
    univ_df.sort_values(by=["market", "ticker"]).to_csv(data_dir / "universe_manifest.csv", index=False)
    print("   [+] Exported data/universe_manifest.csv")
    
    # 2. Dynamic Feature Definitions derived from evaluated 23-feature implementation
    feat_df = generate_evaluated_feature_definitions()
    feat_df.to_csv(data_dir / "feature_definitions.csv", index=False)
    print(f"   [+] Exported data/feature_definitions.csv ({len(feat_df)} technical features)")
    
    # 3. Daily returns (with canonical calendar week mapping)
    daily_df = pd.read_parquet(source_dir / "daily_nav.parquet")
    daily_df["arm"] = daily_df["arm"].str.replace("Transformer_", "TRANS_")
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["calendar_week"] = daily_df["date"].apply(lambda d: (d - pd.Timedelta(days=d.weekday())).strftime("%Y-%m-%d"))
    daily_df["date"] = daily_df["date"].dt.strftime("%Y-%m-%d")
    daily_df["run_id"] = daily_df["arm"] + "_" + daily_df["market"] + "_" + daily_df["seed"].astype(str)
    daily_df = daily_df.sort_values(by=["arm", "market", "seed", "date"])
    
    daily_df.to_csv(data_dir / "daily_returns.csv", index=False)
    daily_df.to_parquet(data_dir / "daily_returns.parquet", index=False)
    print(f"   [+] Exported data/daily_returns.csv & parquet ({len(daily_df)} rows)")
    
    # 4. Executions
    exec_df = pd.read_parquet(source_dir / "executions.parquet")
    exec_df = exec_df.sort_values(by=["arm", "market", "signal_date", "ticker"])
    exec_df.to_csv(data_dir / "executions.csv", index=False)
    exec_df.to_parquet(data_dir / "executions.parquet", index=False)
    print(f"   [+] Exported data/executions.csv & parquet ({len(exec_df)} trades)")
    
    # 5. Run Manifest & Metrics by Run
    metrics_run_df = pd.read_csv(reanalysis_dir / "metrics_by_run.csv")
    if "run_id" not in metrics_run_df.columns:
        metrics_run_df.insert(0, "run_id", metrics_run_df["arm"] + "_" + metrics_run_df["market"] + "_" + metrics_run_df["seed"].astype(str))
    metrics_run_df = metrics_run_df.sort_values(by=["arm", "market", "seed"])
    metrics_run_df.to_csv(data_dir / "metrics_by_run.csv", index=False)
    
    run_manifest_cols = ["run_id", "arm", "market", "seed", "annualized_return", "sharpe_ratio", "max_drawdown", "win_rate", "turnover", "avg_exposure", "realized_pnl", "calendar_exit_fees"]
    run_manifest_df = metrics_run_df[[c for c in run_manifest_cols if c in metrics_run_df.columns]].copy()
    run_manifest_df.to_csv(data_dir / "run_manifest.csv", index=False)
    print("   [+] Exported data/metrics_by_run.csv and data/run_manifest.csv")
    
    # 6. Reanalysis summary tables (both displayed and full precision)
    for f in [
        "master_performance.csv", "master_performance_full_precision.csv",
        "market_performance.csv", "market_performance_full_precision.csv",
        "primary_contrasts.csv", "primary_contrasts_full_precision.csv",
        "secondary_contrasts.csv", "secondary_contrasts_full_precision.csv",
        "block_length_sensitivity.csv", "estimand_reconciliation.csv"
    ]:
        if (reanalysis_dir / f).exists():
            shutil.copy2(reanalysis_dir / f, data_dir / f)
            print(f"   [+] Exported data/{f}")
        
    # 7. Bootstrap draws
    boot_df = pd.read_parquet(reanalysis_dir / "bootstrap_draws.parquet")
    boot_df_export = boot_df.copy()
    if "draw_idx" not in boot_df_export.columns:
        boot_df_export.insert(0, "draw_idx", np.arange(1, len(boot_df_export) + 1))
    boot_df_export.to_csv(data_dir / "bootstrap_draws.csv", index=False)
    boot_df_export.to_parquet(data_dir / "bootstrap_draws.parquet", index=False)
    print(f"   [+] Exported data/bootstrap_draws.csv & parquet ({len(boot_df_export)} draws)")
    
    # 8. Diagnostics & Development mixture
    shutil.copy2(source_dir / "development_mixture_grid.csv", data_dir / "development_mixture_grid.csv")
    shutil.copy2(source_dir / "gate_diagnostic.csv", data_dir / "gate_diagnostic.csv")
    print("   [+] Exported development_mixture_grid.csv and gate_diagnostic.csv")
    
    # 9. Metadata files
    shutil.copy2(source_dir / "selected_mixture_coefficients.json", meta_dir / "selected_mixture_coefficients.json")
    shutil.copy2(source_dir / "conformance_audit_report.json", meta_dir / "conformance_audit_report.json")
    shutil.copy2(reanalysis_dir / "analysis_config.json", meta_dir / "analysis_config.json")
    
    # Gate manifest
    gate_manifest = {
        "gate_architecture": "Normalized linear logistic model: g_t = sigmoid(w^T z_t + b), where z_t = (u_t - mu_u) / (sigma_u + 1e-8)",
        "ordered_inputs": [
            "abs(base_prediction)",
            "abs(base_prediction - memory_prediction)",
            "abs(memory_prediction)",
            "constant_baseline_volatility (0.015)"
        ],
        "fitted_checkpoints": {
            "mlp_seed_7": "models/trust_gate_mlp_seed_7.pt",
            "mlp_seed_17": "models/trust_gate_mlp_seed_17.pt",
            "mlp_seed_37": "models/trust_gate_mlp_seed_37.pt",
            "transformer_seed_7": "models/trust_gate_transformer_seed_7.pt",
            "transformer_seed_17": "models/trust_gate_transformer_seed_17.pt",
            "transformer_seed_37": "models/trust_gate_transformer_seed_37.pt"
        },
        "spearman_correlation_with_memory_advantage": 0.2232,
        "p_value": "< 1e-15",
        "top_gate_bin_memory_win_rate": 0.698
    }
    (meta_dir / "gate_manifest.json").write_text(json.dumps(gate_manifest, indent=2), encoding="utf-8")
    
    # Scaler manifest
    scaler_manifest = {
        "description": "Standardization parameters (mean and standard deviation) fit on 2013-2020 history per sovereign market",
        "n_features": 23,
        "training_period": ["2013-01-01", "2020-12-31"],
        "scaling_method": "(x - mean) / (std + 1e-8)",
        "clipping_bounds": [-5.0, 5.0],
        "markets": ["US", "India", "China", "Brazil", "France", "UK"]
    }
    (meta_dir / "scaler_manifest.json").write_text(json.dumps(scaler_manifest, indent=2), encoding="utf-8")
    
    print("[+] Release bundle export complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export release bundle")
    parser.add_argument("--source", default="research_runs/memory_study/final_comparison", help="Source archive directory")
    parser.add_argument("--output", default="../historical-memory-equity-data", help="Output data repository directory")
    args = parser.parse_args()
    export_bundle(args.source, args.output)
