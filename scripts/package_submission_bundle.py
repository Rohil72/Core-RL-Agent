import os
import shutil
import hashlib
import zipfile
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
pkg_dir = root / "FINAL_SUBMISSION_PACKAGE"
if pkg_dir.exists():
    shutil.rmtree(pkg_dir)
pkg_dir.mkdir(parents=True, exist_ok=True)

# 1. LaTeX Tables
tables_dir = pkg_dir / "manuscript_tables_latex"
tables_dir.mkdir()
for f in (root / "exports/research_defense_bundle/manuscript_tables_latex").glob("*.tex"):
    shutil.copy2(f, tables_dir / f.name)

# 2. Evaluation Matrices
matrices_dir = pkg_dir / "evaluation_matrices"
matrices_dir.mkdir()
for f in (root / "exports/research_defense_bundle/evaluation_matrices").glob("*.*"):
    shutil.copy2(f, matrices_dir / f.name)

# 3. Models
models_dir = pkg_dir / "models"
models_dir.mkdir()
for f in (root / "exports/research_defense_bundle/models").glob("*.pt"):
    shutil.copy2(f, models_dir / f.name)

# 4. Equity Curves & Trade Ledgers
equity_dir = pkg_dir / "equity_curves_and_trades"
equity_dir.mkdir()
raw_equity = root / "paper/internal/evidence/research_defense_extract/raw_experimental_evidence/equity_curves_and_trades"
for f in raw_equity.glob("*.csv"):
    shutil.copy2(f, equity_dir / f.name)

# 5. Paired Returns Bootstrap
bootstrap_dir = pkg_dir / "paired_returns_bootstrap"
bootstrap_dir.mkdir()
raw_boot = root / "paper/internal/evidence/research_defense_extract/raw_experimental_evidence/paired_returns_bootstrap"
for f in raw_boot.glob("*.*"):
    shutil.copy2(f, bootstrap_dir / f.name)

# 6. Audit and Causality Replay
causality_dir = pkg_dir / "causality_replay"
causality_dir.mkdir()
shutil.copy2(root / "paper/internal/evidence/research_defense_extract/raw_experimental_evidence/causality_replay/historical_query_level_causality_replay.csv", causality_dir / "historical_query_level_causality_replay.csv")

split_dir = pkg_dir / "split_boundary_audit"
split_dir.mkdir()
shutil.copy2(root / "paper/internal/evidence/research_defense_extract/raw_experimental_evidence/split_boundary_audit/split_boundary_sample_level_audit.csv", split_dir / "split_boundary_sample_level_audit.csv")

ext_dir = pkg_dir / "external_evaluation_2025_2026"
ext_dir.mkdir()
shutil.copy2(root / "paper/internal/evidence/research_defense_extract/raw_experimental_evidence/external_evaluation_2025_2026/prospective_evaluation_protocol.md", ext_dir / "prospective_evaluation_protocol.md")
shutil.copy2(root / "paper/internal/evidence/research_defense_extract/raw_experimental_evidence/external_evaluation_2025_2026/external_evaluation_2025_2026_equity_curves.csv", ext_dir / "external_evaluation_2025_2026_equity_curves.csv")

# 7. Scalers
scalers_dir = pkg_dir / "provenance_scalers"
scalers_dir.mkdir()
shutil.copy2(root / "paper/internal/evidence/research_defense_extract/raw_experimental_evidence/provenance_scalers/scaler_parameters_23_features.json", scalers_dir / "scaler_parameters_23_features.json")

# 8. Latents
latents_dir = pkg_dir / "latent_space_h1"
latents_dir.mkdir()
raw_latents = root / "paper/internal/evidence/research_defense_extract/raw_experimental_evidence/latent_space_h1"
for f in raw_latents.glob("*.*"):
    shutil.copy2(f, latents_dir / f.name)

# 9. Scripts
scripts_dir = pkg_dir / "scripts"
scripts_dir.mkdir()
shutil.copy2(root / "scripts/run_registered_replication_pipeline.py", scripts_dir / "run_registered_replication_pipeline.py")
shutil.copy2(root / "scripts/verify_reconciled_defense.py", scripts_dir / "verify_reconciled_defense.py")

# 10. Editorial & Evidence Dossiers
dossiers_dir = pkg_dir / "editorial_dossiers"
dossiers_dir.mkdir()
for doc in ["EDITORIAL_CRITICISM_FACT_CHECK_REGISTER_RESOLVED.md", "RESEARCH_LEVEL_EVIDENCE_DOSSIER.md", "VM_RECONSTRUCTION_AND_EVIDENCE_RUNBOOK.md"]:
    p = root / doc
    if p.exists():
        shutil.copy2(p, dossiers_dir / doc)

# 11. README
readme_content = """# Causal Market Memory: Registered Replication & Scientific Defense Package
**Journal:** *Springer Digital Finance*  
**Paper Focus:** Empirical Robustness Audit & Negative Validation of Historical Market Memory  
**Dataset Scope:** 103 Equities across 6 Global Markets (US, India, China, Brazil, France, UK), 2013-2025  

## Directory Structure
- `manuscript_tables_latex/`: All 5 formal publication LaTeX tables (Primary P0-P6, Bootstrap, Ledger, Target Lineage, Universe).
- `models/`: Trained PyTorch GlobalTemporalTransformer encoders (Seeds 7, 17, 37).
- `evaluation_matrices/`: Primary 126-cell matrix, statistical significance tests, and H1 representation diagnostics.
- `equity_curves_and_trades/`: Daily portfolio equity curves (31,563 rows) and trade ledgers (6,528 rows with >=5 day holding periods).
- `paired_returns_bootstrap/`: Panel block bootstrap paired returns and hypothesis tests.
- `audit_and_causality_logs/`: Machine-verifiable causality log (6,250 records, ZERO violations) and split boundary audit (4,830 rows).
- `provenance_scalers/`: Point-in-time scaler parameters fitted on <= 2020-12-31 data.
- `latent_space_h1/`: 128-d PyTorch latents for seeds 7, 17, 37, raw features, and PCA controls.
- `scripts/`: Self-contained replication runner and automated verification script.
- `editorial_dossiers/`: Full fact-check register, research dossier, and execution runbook.

## One-Line Verification
```bash
python scripts/verify_reconciled_defense.py --dir .
```
"""
(pkg_dir / "README.md").write_text(readme_content, encoding="utf-8")

# 12. Checksums
checksums = []
for p in sorted(pkg_dir.rglob("*")):
    if p.is_file():
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        rel = p.relative_to(pkg_dir)
        checksums.append(f"{h}  {rel}")
(pkg_dir / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")

# 13. Create Zip and Tarball
zip_path = root / "FINAL_SUBMISSION_PACKAGE.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in pkg_dir.rglob("*"):
        if p.is_file():
            zf.write(p, arcname=str(p.relative_to(root)))

tar_path = root / "CORE_RL_SPRINGER_DIGITAL_FINANCE_FINAL_SUBMISSION.tar.gz"
with tarfile.open(tar_path, "w:gz") as tf:
    tf.add(pkg_dir, arcname="FINAL_SUBMISSION_PACKAGE")

all_files = [p for p in pkg_dir.rglob("*") if p.is_file()]
print(f"Package Directory: {pkg_dir}")
print(f"Total Files in Package: {len(all_files)}")
print(f"Zip Archive: {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB)")
print(f"Tar.gz Archive: {tar_path} ({tar_path.stat().st_size / 1e6:.2f} MB)")
