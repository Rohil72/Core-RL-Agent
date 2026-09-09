"""Packaging Script for V4 Deep Robustness & Defense Bundle
==========================================================
Packages all 13 robustness analyses and methodological disclosures into:
C:/Users/rohil/Downloads/V4_AUTHORITATIVE_ROBUSTNESS_AND_DEFENSE_BUNDLE/
with standard Unix sha256sum manifest and .zip archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent")
ROBUSTNESS_DIR = PROJECT_ROOT / "paper" / "internal" / "evidence" / "research_defense_extract" / "v4_deep_robustness"
PRIMARY_DIR = PROJECT_ROOT / "paper" / "internal" / "evidence" / "research_defense_extract" / "v4_annual_252_evidence"
DOWNLOADS_DIR = Path("C:/Users/rohil/Downloads")
BUNDLE_NAME = "V4_AUTHORITATIVE_ROBUSTNESS_AND_DEFENSE_BUNDLE"
BUNDLE_DIR = DOWNLOADS_DIR / BUNDLE_NAME

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def package_robustness_bundle():
    print(f"[*] Packaging V4 Deep Robustness Bundle to: {BUNDLE_DIR}")
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy Robustness Evidence Files
    robust_target = BUNDLE_DIR / "robustness_analyses"
    robust_target.mkdir(parents=True, exist_ok=True)
    
    for f in sorted(ROBUSTNESS_DIR.glob("*")):
        if f.is_file():
            shutil.copy2(f, robust_target / f.name)
            print(f"   [+] Robustness: {f.name} ({f.stat().st_size:,} bytes)")
            
    # 2. Copy Primary Matrix and Decision Ledgers
    primary_target = BUNDLE_DIR / "primary_v4_evidence"
    primary_target.mkdir(parents=True, exist_ok=True)
    for p_name in ["v4_primary_systems_126_cell_matrix.csv", "v4_trade_ledgers_p0_p6.csv", "v4_query_neighbor_decision_ledger.csv", "v4_experiment_metadata.json"]:
        p_file = PRIMARY_DIR / p_name
        if p_file.exists():
            shutil.copy2(p_file, primary_target / p_name)
            print(f"   [+] Primary: {p_name} ({p_file.stat().st_size:,} bytes)")
            
    # 3. Copy Methodological Disclosure & Protocol to Root
    proto_src = ROBUSTNESS_DIR / "METHODOLOGICAL_DISCLOSURE_AND_PROSPECTIVE_PROTOCOL.md"
    if proto_src.exists():
        shutil.copy2(proto_src, BUNDLE_DIR / "METHODOLOGICAL_DISCLOSURE_AND_PROSPECTIVE_PROTOCOL.md")
        
    # 4. Copy Execution Scripts
    scripts_target = BUNDLE_DIR / "scripts"
    scripts_target.mkdir(parents=True, exist_ok=True)
    for s_name in ["run_v4_annual_252_patch_pipeline.py", "run_v4_deep_robustness_suite.py"]:
        s_file = PROJECT_ROOT / "scripts" / s_name
        if s_file.exists():
            shutil.copy2(s_file, scripts_target / s_name)
            
    # 5. Provenance Metadata
    manifest_data = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bundle_name": BUNDLE_NAME,
        "bundle_version": "V4-Authoritative-Robustness-v1.0",
        "python_version": sys.version,
        "platform": platform.platform(),
        "included_analyses": [
            "1. Multi-block-length panel bootstrap (L = 5, 10, 21, 63)",
            "2. Synchronized-date and market-clustered inference",
            "3. Fully liquidated terminal-equity metrics",
            "4. Duration distribution and Kaplan-Meier survival curves",
            "5. 1,000 random-ranking Monte Carlo runs (empirical null)",
            "6. Fixed-horizon exit comparisons (H = 5, 21, 63)",
            "7. Trailing-stop sensitivity sweep (1.5x, 2.0x, 2.5x, 3.0x ATR, static 10%, 12%)",
            "8. P0/P3/P4 timestamp alignment and return correlations",
            "9. Full 25-neighbor export",
            "10. Memory age distribution, Gini hubness, and HHI concentration",
            "11. SVD representation diagnostics and effective rank",
            "12. Detectable effect size (MDES) and power calculations",
            "13. Factor and benchmark exposure analysis (P6 market, P4 momentum)",
            "14. Methodological disclosure and prospective testing protocol",
        ]
    }
    with open(BUNDLE_DIR / "bundle_provenance.json", "w") as f:
        json.dump(manifest_data, f, indent=2)

    # 6. Standard Unix-style SHA256SUMS.txt
    checksum_lines = []
    for p in sorted(BUNDLE_DIR.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            rel_p = p.relative_to(BUNDLE_DIR).as_posix()
            h = sha256_file(p)
            checksum_lines.append(f"{h}  {rel_p}")
            
    sums_file = BUNDLE_DIR / "SHA256SUMS.txt"
    sums_file.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"\n[+] Generated SHA256SUMS.txt ({len(checksum_lines)} files)")

    # 7. Create Zip Archive
    zip_target = DOWNLOADS_DIR / f"{BUNDLE_NAME}.zip"
    shutil.make_archive(str(BUNDLE_DIR), "zip", BUNDLE_DIR)
    print(f"[+] Created Zip Archive: {zip_target} ({zip_target.stat().st_size:,} bytes)")

if __name__ == "__main__":
    package_robustness_bundle()
