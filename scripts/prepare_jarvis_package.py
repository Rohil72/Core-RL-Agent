"""Prepare a reproducible package for running experiments on Jarvis Lab.

What this does:
- Validates local precomputed data and model artifacts required for training.
- Assembles a minimal package under dist/jarvis_package/ containing code, configs, a Dockerfile template, and a small requirements.txt.
- Produces a tarball ready to upload to a remote host or build inside a remote registry.

Usage:
  python .\scripts\prepare_jarvis_package.py --include-data [--model-path models\\candidate_scorer.pt]

Notes:
- The script does NOT upload anything. It only prepares a self-contained archive and checks.
- Edit configs/jarvis_lab.yaml to configure remote job parameters (registry, image name, GPU type).
"""
from __future__ import annotations

import argparse
import os
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "jarvis_package"
INCLUDE = ["src", "configs", "scripts", "trainers.py", "README.md"]

DEFAULT_MODEL = Path("models") / "candidate_scorer.pt"

REQUIREMENTS = """
numpy
pandas
pyyaml
tqdm
pyarrow
yfinance
torch
"""

DOCKERFILE = """
# Use an official PyTorch runtime with CUDA (adjust tag to match Jarvis CUDA version)
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Copy repository into container
COPY . /workspace

# Install minimal Python dependencies
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Default command: run trainer script (override at container runtime)
CMD ["python", "trainers.py", "--data-dir", "data/precomputed", "--device", "cuda"]
"""


def _copy_tree(src: Path, dst: Path, include_patterns=None):
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    for root, dirs, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
        for f in files:
            srcf = Path(root) / f
            if include_patterns:
                if not any(p in str(srcf) for p in include_patterns):
                    continue
            dstf = dst / rel_root / f
            dstf.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(srcf, dstf)


def prepare(output_dir: Path, include_data: bool, model_path: Path | None):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Copying code files...")
    for item in INCLUDE:
        src = ROOT / item
        if not src.exists():
            # skip optional files
            continue
        dst = output_dir / item
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # write requirements.txt and Dockerfile
    (output_dir / "requirements.txt").write_text(REQUIREMENTS.strip() + "\n")
    (output_dir / "Dockerfile").write_text(DOCKERFILE.strip() + "\n")

    # include model if requested
    if model_path is None:
        model_path = DEFAULT_MODEL
    if model_path.exists():
        print(f"Including model artifact: {model_path}")
        dst_model = output_dir / model_path
        dst_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, dst_model)
    else:
        print(f"Warning: model not found at {model_path} — package will not include scorer artifact.")

    # include precomputed data optionally (warning: can be large)
    if include_data:
        data_dir = ROOT / "data" / "precomputed"
        if data_dir.exists():
            print("Including precomputed data (may be large)...")
            dst_data = output_dir / "data" / "precomputed"
            dst_data.parent.mkdir(parents=True, exist_ok=True)
            # copy parquet files only
            for f in data_dir.glob("*.parquet"):
                shutil.copy2(f, dst_data / f.name)
        else:
            print("No precomputed data found to include.")

    # create tarball
    tar_path = ROOT / "dist" / "jarvis_package.tar.gz"
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating tarball {tar_path}...")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(output_dir, arcname="jarvis_package")

    print("Package prepared.")
    print("Contents:")
    for p in sorted(output_dir.rglob("*")):
        print(" ", p.relative_to(output_dir))
    print("Produced:", tar_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(DIST))
    p.add_argument("--include-data", action="store_true")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL))
    args = p.parse_args()
    prepare(Path(args.output_dir), include_data=args.include_data, model_path=Path(args.model_path))
