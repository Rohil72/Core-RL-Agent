# VM Execution, Evidence Packaging, and Download Runbook

This runbook provides copy-paste commands to execute the reconstruction pipeline, run causality audits, generate matched baseline comparisons (P0–P6), package tamper-sealed research evidence, create downloadable `.tar.gz` archives, and verify data integrity for academic or institutional defense.

---

## 1. VM Environment Preparation

### 1.1 Pull or Sync Repository
```bash
cd ~
git clone https://github.com/Rohil72/Core-RL-Agent.git  # or cd Core-RL-Agent && git fetch && git checkout <branch>
cd Core-RL-Agent
```

### 1.2 Python Virtual Environment & CUDA PyTorch Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip, wheel, and setuptools
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (adjust cu121/cu118 based on nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install core scientific and evaluation dependencies
pip install -r requirements.txt
pip install pytest pyyaml pandas numpy scipy scikit-learn
```

### 1.3 Pre-flight Verification (GPU & Test Suite)
```bash
# Check GPU recognition and VRAM
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB' if torch.cuda.is_available() else 'No GPU')"

# Run the 30 core causality, representation, primary systems, and extractor tests
python3 -m pytest tests/test_causality_invariants.py \
                 tests/test_primary_systems.py \
                 tests/test_representation_h1.py \
                 tests/test_statistical_bootstrap.py \
                 tests/test_universe_ledger.py \
                 tests/test_research_defense_extractor.py -v
```

---

## 2. Running Reconstruction & Evidence Pipelines

You can run individual stages or execute the end-to-end master reconstruction.

### Option A: End-to-End Master Pipeline (Recommended)
Executes causality audit, universe ledger, H1 representation diagnostics, primary systems comparison, and generates the sealed evidence bundle:
```bash
python3 scripts/run_reconstruction_v1.py --stage all --device cuda
```
*(If running without a GPU or in a test VM, use `--device cpu`)*

### Option B: Stage-by-Stage Execution

#### Stage 1: Causality and 11-Target Lineage Audit
Verifies target maturation rules and certifies 252-session target isolation:
```bash
python3 scripts/run_reconstruction_v1.py --stage audit
```
*Output: `reports/reconstruction_v1/audit/target_lineage_and_causality_audit.json`*

#### Stage 2: Multi-Market Universe Ledger
Builds the 6-market security inventory (108 requested -> 103 available -> 5 excluded):
```bash
python3 scripts/run_reconstruction_v1.py --stage universe
```
*Output: `reports/reconstruction_v1/manifests/universe_ledger.json`*

#### Stage 3 & 4: H1 Representation Diagnostics
Runs Linear CKA, seed-to-seed kNN Jaccard overlap, outcome homogeneity, and PCA controls:
```bash
python3 scripts/run_reconstruction_v1.py --stage representation
```
*Output: `reports/reconstruction_v1/representation/h1_representation_diagnostics.json`*

#### Stage 5: Independent Replay Audit
Recomputes Gaussian weights, trade P&L, and slippage calculations to prove zero calculation drift:
```bash
python3 scripts/replay_reconstruction_audit.py --run-root reports/reconstruction_v1
```
*Output: `reports/reconstruction_v1/replay_audit_summary.json`*

---

## 3. Automated Research Defense Evidence Packaging

The extractor script captures Git commit provenance, hardware signatures, LaTeX tables, and creates a cryptographically sealed package with SHA-256 checksums and an offline verification script.

### 3.1 Standard Reviewer Package (Lightweight: JSON, CSV, LaTeX, YAML, Markdown)
Fast to download (~10MB–50MB), containing all tables, ledgers, metrics, and causality certificates:
```bash
mkdir -p exports
python3 scripts/extract_research_defense_package.py \
    --run-root reports/reconstruction_v1 \
    --output exports/research_defense_bundle.tar.gz \
    --format all \
    --profile reviewer \
    --title "Causal Market Memory Reconstruction" \
    --verify
```

### 3.2 Full Institutional Package (Includes Parquet Signal & Trade Tables)
Includes all high-resolution parquet tables, detailed logs, and full artifact traces:
```bash
python3 scripts/extract_research_defense_package.py \
    --run-root reports/reconstruction_v1 \
    --output exports/full_research_evidence.tar.gz \
    --format tar.gz \
    --profile full \
    --verify
```

---

## 4. Manual / Granular TAR Archiving Commands

Use these commands on the VM to create targeted `.tar.gz` archives for selective downloading:

### 4.1 Bundle 1: Core Audits, Configurations, and Publication Artifacts (Smallest, ~5–15 MB)
```bash
mkdir -p ~/transfers

tar -czvf ~/transfers/core_audit_and_latex.tar.gz \
    configs/reconstruction_v1.yaml \
    MANUSCRIPT_HARDENING_EVIDENCE_V3.md \
    RECOVERED_CREDIBILITY_EVIDENCE.md \
    reports/reconstruction_v1/manifests/ \
    reports/reconstruction_v1/audit/ \
    reports/reconstruction_v1/representation/ \
    reports/reconstruction_v1/*.json \
    reports/reconstruction_v1/research_defense_bundle/
```

### 4.2 Bundle 2: Complete Trade Ledgers and Metric Results (~50–200 MB)
```bash
tar -czvf ~/transfers/trade_and_metrics_evidence.tar.gz \
    --exclude="*.parquet" \
    reports/reconstruction_v1/ \
    reports/final_testbed/
```

### 4.3 Bundle 3: Full High-Resolution Signal & Neighbour Parquet Ledgers (1 GB+)
```bash
tar -czvf ~/transfers/full_parquet_signals_ledgers.tar.gz \
    $(find reports/ -name "*.parquet" -o -name "*neighbour*.parquet" -o -name "*trade*.csv")
```

### 4.4 Bundle 4: All-in-One Comprehensive Research Vault (Everything)
```bash
tar -czvf ~/transfers/complete_reconstruction_vault_$(date +%Y%m%d_%H%M%S).tar.gz \
    configs/ \
    scripts/ \
    src/ \
    tests/ \
    reports/reconstruction_v1/ \
    exports/
```

### 4.5 Generate Checksums for All Created Archives
Run this on the VM to produce a verification hash list before downloading:
```bash
cd ~/transfers
sha256sum *.tar.gz > SHA256SUMS.txt
cat SHA256SUMS.txt
```

---

## 5. Downloading Evidence from VM to Local Machine

Run these commands on your **local machine** (PowerShell or Terminal) to download the archives.

### Method 1: Using `scp` (Secure Copy)
Replace `<vm-ip>` and `<user>` with your VM credentials:
```bash
# Download the sealed research defense package
scp -i ~/.ssh/id_rsa user@<vm-ip>:~/Core-RL-Agent/exports/research_defense_bundle.tar.gz .

# Download the granular archives and checksums
scp -i ~/.ssh/id_rsa user@<vm-ip>:~/transfers/*.tar.gz .
scp -i ~/.ssh/id_rsa user@<vm-ip>:~/transfers/SHA256SUMS.txt .
```

### Method 2: Using `rsync` (Resumeable & Progress Bar)
```bash
rsync -avzP -e "ssh -i ~/.ssh/id_rsa" user@<vm-ip>:~/transfers/ ./downloaded_evidence/
```

### Method 3: Simple On-Demand HTTP Server (If SSH ports are restricted)
On the VM:
```bash
cd ~/transfers
python3 -m http.server 8080
```
On your local browser, navigate to:
`http://<vm-ip>:8080/` and click any `.tar.gz` to download directly.
*(Remember to Ctrl+C the HTTP server once downloaded)*

---

## 6. Local Verification and Evidence Inspection

Once downloaded to your local machine:

### 6.1 Verify Archive Checksums
On Linux/macOS:
```bash
sha256sum -c SHA256SUMS.txt
```
On Windows PowerShell:
```powershell
Get-FileHash -Algorithm SHA256 *.tar.gz | Format-Table -AutoSize
Get-Content SHA256SUMS.txt
```

### 6.2 Extract and Verify Sealed Research Defense Bundle
```bash
tar -xzvf research_defense_bundle.tar.gz
cd research_defense_bundle

# Run the zero-dependency offline validator
python verify_bundle.py
```
Expected output:
```text
============================================================
RESEARCH DEFENSE BUNDLE INTEGRITY CHECK
============================================================
Scanned files: 18
Verified:      18
Mismatches:    0
Missing:       0
STATUS:        PASSED (All cryptographic signatures valid)
============================================================
```

### 6.3 What to Inspect in the Extracted Package

| File | Purpose for Editorial Defense |
|---|---|
| `latex/tab_universe_retention.tex` | Publication LaTeX table explaining 108 requested vs 103 available securities. |
| `latex/tab_target_lineage.tex` | Publication LaTeX table certifying 252-session target isolation and maturity rules. |
| `latex/tab_primary_systems.tex` | LaTeX comparison of Primary Systems P0 through P6 under identical constraints. |
| `latex/tab_statistical_bootstrap.tex` | Moving-block bootstrap 95% CI, p-values, Holm, and FDR corrections. |
| `metadata/provenance.json` | Captured Git commit hash, branch, dirty diff, OS, CUDA/cuDNN version, and GPU VRAM. |
| `metadata/environment_packages.txt` | Complete `pip freeze` dependency lock. |
| `audit/target_lineage_and_causality_audit.json` | Machine-evaluated results for all 14 causality invariants. |
| `summary/executive_defense_summary.md` | Executive synthesis of methods, gates, limitations, and empirical findings. |
| `checksums.sha256` | SHA-256 fingerprint for every single individual file in the package. |
