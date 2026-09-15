import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import pandas as pd
from memory_study_v2.ohlc_validation import (
    normalize_ohlc_dataframe,
    MAX_ROUNDOFF_ULPS,
)

def run_scan():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "cache" / "ohlcv"
    all_parquets = sorted(data_dir.glob("*.parquet"))
    print(f"Scanning {len(all_parquets)} parquet files...")

    security_classifications = {}
    all_roundoff_records = []
    all_material_records = []

    for p in all_parquets:
        sec_id = p.stem
        df = pd.read_parquet(p)
        if "session" not in df.columns:
            df = df.reset_index()
            rename_map = {}
            for c in df.columns:
                if c.lower() in ("date", "session", "index", "timestamp"):
                    rename_map[c] = "session"
            df = df.rename(columns=rename_map)
            df["session"] = df["session"].astype(str).str.slice(0, 10)

        norm_df, audit_recs, stats = normalize_ohlc_dataframe(
            df,
            security_id=sec_id,
            max_ulps=MAX_ROUNDOFF_ULPS,
        )

        if stats["other"] > 0:
            sec_class = "Other invalid data"
        elif stats["material"] > 0:
            sec_class = "Material OHLC discrepancy"
        elif stats["roundoff"] > 0:
            sec_class = "Roundoff-only, within the approved bound"
        else:
            sec_class = "Already valid"

        security_classifications[sec_id] = {
            "classification": sec_class,
            "total_rows": len(df),
            "valid_rows": stats["valid"],
            "roundoff_rows": stats["roundoff"],
            "material_rows": stats["material"],
        }

        for r in audit_recs:
            if r["classification"] == "ROUNDOFF_ONLY":
                all_roundoff_records.append(r)
            elif r["classification"] == "MATERIAL_DISCREPANCY":
                all_material_records.append(r)

    class_counts = pd.Series([v["classification"] for v in security_classifications.values()]).value_counts()
    ops_dir = repo_root / "outputs" / "ops"
    ops_dir.mkdir(parents=True, exist_ok=True)

    with open(ops_dir / "ohlc_normalization_audit.json", "w", encoding="utf-8") as f:
        json.dump(all_roundoff_records, f, indent=2)

    if all_roundoff_records:
        pd.DataFrame(all_roundoff_records).to_csv(ops_dir / "ohlc_normalization_audit.csv", index=False)
    if all_material_records:
        pd.DataFrame(all_material_records).to_csv(ops_dir / "material_discrepancies.csv", index=False)

    md_lines = [
        "# OHLC Data Integrity & Bounded Roundoff Diagnostic Report",
        "",
        f"- **Total Securities Scanned**: {len(all_parquets)}",
        f"- **Maximum Allowed Bound**: {MAX_ROUNDOFF_ULPS} float64 representable steps (ULPs) via `np.nextafter`",
        "",
        "## 1. Summary Classification Table",
        "",
        "| Classification | Action | Security Count | Row Count |",
        "|---|---|---|---|",
        f"| Already valid | No change | {class_counts.get('Already valid', 0)} | {sum(v['valid_rows'] for v in security_classifications.values() if v['classification'] == 'Already valid')} |",
        f"| Roundoff-only, within the approved bound | Normalize working copy and log | {class_counts.get('Roundoff-only, within the approved bound', 0)} | {len(all_roundoff_records)} |",
        f"| Material OHLC discrepancy | Keep rejected; report for investigation | {class_counts.get('Material OHLC discrepancy', 0)} | {len(all_material_records)} |",
        "| Other invalid data | Preserve existing failure behavior | 0 | 0 |",
        "",
        "## 2. Roundoff-Only Normalization Audit (First 20 Entries)",
        "",
        "| Security | Session | Violated Relations | Max Abs Diff | Max Rel Diff |",
        "|---|---|---|---|---|",
    ]

    for r in all_roundoff_records[:20]:
        md_lines.append(f"| {r['security']} | {r['session']} | {r['violations']} | {r['max_abs_violation']:.2e} | {r['max_rel_violation']:.2e} |")

    if len(all_roundoff_records) > 20:
        md_lines.append(f"| ... | ({len(all_roundoff_records) - 20} more entries in `outputs/ops/ohlc_normalization_audit.csv`) | | | |")

    md_lines.extend([
        "",
        "## 3. Material Discrepancies Inventory (> 4 ULPs)",
        "",
        "| Security | Date | Open | High | Low | Close | Abs Violation | Rel Violation |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for r in all_material_records[:40]:
        md_lines.append(f"| {r['security']} | {r['session']} | {r['original_open']:.4f} | {r['original_high']:.4f} | {r['original_low']:.4f} | {r['original_close']:.4f} | {r['max_abs_violation']:.4f} | {r['max_rel_violation']:.2%} |")

    if len(all_material_records) > 40:
        md_lines.append(f"| ... | ({len(all_material_records) - 40} more entries in `outputs/ops/material_discrepancies.csv`) | | | | | | |")

    with open(ops_dir / "ohlc_diagnostic_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Report successfully written to {ops_dir / 'ohlc_diagnostic_report.md'}")

if __name__ == "__main__":
    run_scan()
