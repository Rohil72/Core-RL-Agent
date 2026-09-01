import json
from pathlib import Path

base = Path(r"C:\Users\rohil\Downloads\updated results")

print("=" * 60)
print("UPDATED RESULTS AUDIT & VERIFICATION REPORT")
print("=" * 60)

# 1. Environment
env_p = base / "reports" / "reconstruction_v1" / "environment_hardware.json"
if env_p.exists():
    with open(env_p, "r", encoding="utf-8") as f:
        env = json.load(f)
        print("\n[1] PLATFORM & ACCELERATOR:")
        print(f"  Device: {env.get('devices', [{}])[0].get('name')} ({env.get('devices', [{}])[0].get('total_memory_gb')} GB VRAM)")
        print(f"  PyTorch: {env.get('torch_version')} | CUDA: {env.get('torch_cuda_version')} | cuDNN: {env.get('cudnn_version')}")

# 2. Universe
univ_p = base / "reports" / "reconstruction_v1" / "manifests" / "universe_ledger.json"
if not univ_p.exists():
    univ_p = base / "exports" / "research_defense_bundle" / "causality_and_data_integrity" / "universe_retention_ledger.json"
if univ_p.exists():
    with open(univ_p, "r", encoding="utf-8") as f:
        u = json.load(f)
        print("\n[2] UNIVERSE ACCOUNTING (103 TICKERS):")
        print(f"  Totals: Requested={u.get('totals', {}).get('requested')}, Available={u.get('totals', {}).get('available')}, Excluded={u.get('totals', {}).get('excluded')}")
        for m, data in u.get("by_market", {}).items():
            print(f"  - {m}: {data.get('available_count')}/{data.get('requested_count')} available (Exclusions: {list(data.get('excluded_tickers', {}).keys())})")

# 3. Causality Invariant Audit
causal_p = base / "reports" / "reconstruction_v1" / "audit" / "target_lineage_and_causality_audit.json"
if causal_p.exists():
    with open(causal_p, "r", encoding="utf-8") as f:
        c = json.load(f)
        print("\n[3] CAUSALITY & TARGET ISOLATION:")
        print(f"  All Invariants Pass: {c.get('all_invariants_pass')}")
        print(f"  252 Target Isolated: {c.get('target_252_isolation', {}).get('status')} (Violations: {c.get('target_252_isolation', {}).get('violations')})")
        print(f"  Maturity Invariant: {c.get('invariants', {}).get('memory_field_maturity', {}).get('status')}")
        print(f"  Same-Ticker Exclusion: {c.get('invariants', {}).get('same_ticker_exclusion', {}).get('status')}")
        print(f"  Temporal Separation (>=21d): {c.get('invariants', {}).get('temporal_separation', {}).get('status')}")

# 4. Primary Systems P0-P6
p0_p = base / "reports" / "reconstruction_v1" / "research_defense_bundle" / "evaluation_matrices" / "primary_systems_p0_p6.json"
if p0_p.exists():
    with open(p0_p, "r", encoding="utf-8") as f:
        p0 = json.load(f)
        print("\n[4] PRIMARY SYSTEMS MATRIX (P0-P6):")
        print(f"  {'System':<6} | {'Name':<36} | {'Return':<8} | {'Sharpe':<6} | {'MaxDD':<8} | {'WinRate':<7}")
        print("  " + "-" * 78)
        for row in p0:
            print(f"  {row['System']:<6} | {row['Name']:<36} | {row['Total Return']:<8.2%} | {row['Sharpe']:<6.3f} | {row['Max Drawdown']:<8.2%} | {row['Win Rate']:<7.1%}")

# 5. Bootstrap Tests
stat_p = base / "reports" / "reconstruction_v1" / "research_defense_bundle" / "evaluation_matrices" / "statistical_significance_tests.json"
if stat_p.exists():
    with open(stat_p, "r", encoding="utf-8") as f:
        st = json.load(f)
        print("\n[5] MOVING-BLOCK BOOTSTRAP SIGNIFICANCE (1,000 resamples, 21d blocks):")
        print(f"  {'Comparison':<36} | {'dSharpe':<8} | {'95% CI':<16} | {'p_Holm':<8} | {'q_FDR':<8}")
        print("  " + "-" * 82)
        for row in st:
            ci = f"[{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}]"
            print(f"  {row['comparison']:<36} | {row['point_estimate']:<+8.3f} | {ci:<16} | {row['p_holm']:<8.4f} | {row['q_fdr']:<8.5f}")

# 6. H1 Representation
h1_p = base / "reports" / "reconstruction_v1" / "representation" / "h1_representation_diagnostics.json"
if h1_p.exists():
    with open(h1_p, "r", encoding="utf-8") as f:
        h1 = json.load(f)
        print("\n[6] H1 REPRESENTATION INVARIANCE DIAGNOSTICS:")
        print(f"  Linear CKA (Seeds 7 vs 17): {h1.get('cka_seeds_7_vs_17'):.4f}")
        print(f"  Linear CKA (Seeds 7 vs 37): {h1.get('cka_seeds_7_vs_37'):.4f}")
        print(f"  kNN Overlap (k=25):         {h1.get('knn_overlap_seeds_7_vs_17'):.4f}")
        print(f"  Neighbour MAE (Learned):    {h1.get('neighbour_outcome_mae_learned'):.4f}")
        print(f"  Neighbour MAE (Raw Control):{h1.get('neighbour_outcome_mae_raw_control'):.4f}")
        print(f"  Neighbour MAE (PCA Control):{h1.get('neighbour_outcome_mae_pca_control'):.4f}")
        print(f"  Nuisance Ticker Decodability: {h1.get('nuisance_ticker_accuracy'):.2%}")

print("\n" + "=" * 60)
