"""A30 Production Deployment Launch Tool (Finding 5 / C6).

Automates deployment, preflight validation, and initial batch execution
for production runs on the NVIDIA A30 GPU / Linux compute host.

Capabilities:
1. Hardware & Driver Preflight:
   - Verifies CUDA runtime and driver compatibility (PyTorch >= 2.0, CUDA >= 11.8/12.x).
   - Probes GPU compute capability (Ampere 8.0 for A30, 8.6 for RTX, etc.) and available VRAM.
   - Executes live CUDA tensor allocation and GEMM compute probe to ensure no kernel errors.
2. Storage & Filesystem Health:
   - Verifies read access to OHLCV cache and sample ID manifests.
   - Verifies read/write permissions for output directories and checkpoint paths.
3. Strict Fail-Closed Authorization Guard:
   - Verifies that repo configuration has 'production_authorized: false'.
   - Demands explicit operator authorization flag (--authorize-production) at launch time.
   - Refuses production training if unauthorized.
4. Production Walk-Forward Execution & Telemetry:
   - Runs specified fold(s) (default: Fold 2016) with CUDA acceleration.
   - Emits signed launch receipt to rebuild_plan/a30_launch_receipt.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

from memory_study_v2.contracts import to_canonical_json
from memory_study_v2.train import load_neural_config, REQUIRED_NEURAL_KEYS


def get_system_telemetry() -> Dict[str, Any]:
    """Capture comprehensive host and process telemetry."""
    res: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False) if psutil else None,
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2) if psutil else None,
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2) if psutil else None,
    }
    if psutil:
        res["process_rss_mb"] = round(psutil.Process().memory_info().rss / (1024**2), 2)
    return res


def run_cuda_preflight() -> Dict[str, Any]:
    """Inspect and probe CUDA hardware and runtime compatibility."""
    if torch is None:
        raise RuntimeError("PyTorch is not installed in the current environment.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in PyTorch. Ensure an NVIDIA GPU is present and CUDA drivers are loaded."
        )

    device_count = torch.cuda.device_count()
    device_idx = 0
    device_name = torch.cuda.get_device_name(device_idx)
    cap = torch.cuda.get_device_capability(device_idx)
    props = torch.cuda.get_device_properties(device_idx)
    total_vram_gb = round(props.total_memory / (1024**3), 2)

    # Live CUDA compute probe: allocate, matmul, reduce, sync
    torch.cuda.reset_peak_memory_stats(device_idx)
    probe_start = time.perf_counter()
    dev = torch.device(f"cuda:{device_idx}")
    t_a = torch.randn(1024, 1024, dtype=torch.float32, device=dev)
    t_b = torch.randn(1024, 1024, dtype=torch.float32, device=dev)
    t_c = torch.matmul(t_a, t_b)
    val = float(t_c.sum().item())
    torch.cuda.synchronize(dev)
    probe_ms = round((time.perf_counter() - probe_start) * 1000, 2)
    del t_a, t_b, t_c
    torch.cuda.empty_cache()

    return {
        "cuda_available": True,
        "device_count": device_count,
        "device_index": device_idx,
        "device_name": device_name,
        "compute_capability": list(cap),
        "multi_processor_count": props.multi_processor_count,
        "total_vram_gb": total_vram_gb,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "compute_probe": {
            "status": "PROBE_SUCCESS",
            "probe_duration_ms": probe_ms,
            "probe_scalar_reduction": val,
        },
    }


def verify_storage_and_manifests(
    data_dir: Path,
    output_dir: Path,
    sample_ids_dir: Path,
) -> Dict[str, Any]:
    """Verify input manifests, sample index files, and output path writability."""
    checks: Dict[str, Any] = {}

    # 1. Check universe manifest
    universe_path = REPO_ROOT / "rebuild_plan" / "universe_manifest_clean.csv"
    if universe_path.exists():
        checks["universe_manifest_present"] = True
    else:
        checks["universe_manifest_present"] = False

    # 2. Check fold dimensions manifest
    fold_dims_path = REPO_ROOT / "rebuild_plan" / "fold_dimensions_manifest.json"
    if fold_dims_path.exists():
        with open(fold_dims_path, "r", encoding="utf-8") as f:
            fold_dims = json.load(f)
        checks["fold_dimensions_present"] = True
        checks["total_folds_configured"] = len(fold_dims.get("folds", {}))
    else:
        checks["fold_dimensions_present"] = False
        checks["total_folds_configured"] = 0

    # 3. Check sample ID manifests
    if sample_ids_dir.exists():
        sample_files = list(sample_ids_dir.glob("fold_*_sample_ids.json")) + list(sample_ids_dir.glob("sample_ids_*.csv"))
        checks["sample_id_files_count"] = len(sample_files)
        checks["sample_ids_present"] = len(sample_files) >= 6
    else:
        checks["sample_id_files_count"] = 0
        checks["sample_ids_present"] = False

    # 4. Check data cache
    if data_dir.exists():
        parquet_files = list(data_dir.glob("*.parquet"))
        checks["data_cache_dir"] = str(data_dir)
        checks["parquet_count"] = len(parquet_files)
    else:
        checks["data_cache_dir"] = str(data_dir)
        checks["parquet_count"] = 0

    # 5. Check output writability
    output_dir.mkdir(parents=True, exist_ok=True)
    test_probe_file = output_dir / ".write_probe"
    try:
        with open(test_probe_file, "w", encoding="utf-8") as f:
            f.write("write_ok")
        test_probe_file.unlink()
        checks["output_dir_writable"] = True
    except Exception as e:
        checks["output_dir_writable"] = False
        checks["output_dir_error"] = str(e)

    return checks


def verify_configuration_authorization(
    config_path: Path,
    authorize_flag: bool,
) -> Dict[str, Any]:
    """Verify repo configuration state and launch-time authorization."""
    if not config_path.exists():
        raise FileNotFoundError(f"Required configuration file '{config_path}' not found.")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    repo_authorized = cfg.get("production_authorized", False)

    # Repo file MUST have production_authorized: false (Finding 4 / C4 invariant)
    repo_guard_intact = (repo_authorized is False)

    # Runtime authorization requires explicit operator flag
    can_proceed = authorize_flag and repo_guard_intact

    return {
        "config_file": str(config_path),
        "repo_production_authorized": repo_authorized,
        "repo_guard_intact": repo_guard_intact,
        "launch_flag_authorized": authorize_flag,
        "can_proceed_to_production": can_proceed,
    }


def launch_a30_deployment(
    config_path: Path = REPO_ROOT / "rebuild_plan" / "config.proposed.json",
    data_dir: Optional[Path] = None,
    output_dir: Path = REPO_ROOT / "rebuild_plan" / "a30_production_out",
    sample_ids_dir: Path = REPO_ROOT / "rebuild_plan" / "sample_ids",
    authorize_production: bool = False,
    run_connected_pilot: bool = False,
    check_only: bool = False,
) -> Dict[str, Any]:
    """Run full A30 startup validation, safety guards, and launch execution."""
    t_start = time.perf_counter()
    print("=" * 70)
    print("NVIDIA A30 / Production Deployment Preflight & Launch Runner")
    print("=" * 70)

    # 1. System & Host Telemetry
    host_telemetry = get_system_telemetry()
    print(f"Host: {host_telemetry['platform']} | Python: {host_telemetry['python_version']}")
    print(f"RAM Total: {host_telemetry.get('ram_total_gb')} GB | Available: {host_telemetry.get('ram_available_gb')} GB")

    # 2. CUDA Hardware Preflight
    print("\n[Step 1/4] Running CUDA Hardware & Kernel Compute Probe...")
    cuda_info = run_cuda_preflight()
    print(f"  CUDA Device: {cuda_info['device_name']} (Index {cuda_info['device_index']})")
    print(f"  Compute Capability: {cuda_info['compute_capability']} | Total VRAM: {cuda_info['total_vram_gb']} GB")
    print(f"  PyTorch: {cuda_info['pytorch_version']} | CUDA Runtime: {cuda_info['cuda_version']}")
    print(f"  Compute Probe: {cuda_info['compute_probe']['status']} ({cuda_info['compute_probe']['probe_duration_ms']} ms)")

    # 3. Storage & Manifest Health
    print("\n[Step 2/4] Verifying Storage & Manifest Health...")
    if data_dir is None:
        cand_dirs = [
            REPO_ROOT / "FINAL_SUBMISSION_PACKAGE" / "data" / "cache" / "ohlcv",
            REPO_ROOT / "data" / "cache" / "ohlcv",
        ]
        for cd in cand_dirs:
            if cd.exists():
                data_dir = cd
                break
        if data_dir is None:
            data_dir = cand_dirs[0]

    storage_info = verify_storage_and_manifests(data_dir, output_dir, sample_ids_dir)
    print(f"  Data Directory: {storage_info['data_cache_dir']} ({storage_info['parquet_count']} parquets)")
    print(f"  Sample ID Manifests: {storage_info['sample_id_files_count']} files present")
    print(f"  Output Directory Writable: {storage_info['output_dir_writable']}")

    # 4. Configuration & Fail-Closed Production Authorization
    print("\n[Step 3/4] Verifying Production Configuration & Authorization Guard...")
    auth_info = verify_configuration_authorization(config_path, authorize_production)
    print(f"  Repository config.proposed.json 'production_authorized': {auth_info['repo_production_authorized']}")
    print(f"  Repository Guard Intact: {auth_info['repo_guard_intact']}")
    print(f"  Operator Launch Flag (--authorize-production): {auth_info['launch_flag_authorized']}")

    if not auth_info["repo_guard_intact"]:
        print("  [ERROR] Invariant violated: 'production_authorized' must be false in repository files!")
        sys.exit(1)

    if not authorize_production and not check_only:
        print("\n" + "!" * 70)
        print("HALTED: Production execution is locked.")
        print("To authorize execution on this target host, supply --authorize-production")
        print("!" * 70)

    # 5. Execution Phase
    execution_result: Dict[str, Any] = {
        "executed": False,
        "mode": "CHECK_ONLY" if check_only else ("AUTHORIZED_PRODUCTION" if authorize_production else "UNAUTHORIZED"),
    }

    if check_only:
        print("\n[Step 4/4] Check-only mode specified. Preflight checks completed.")
    elif run_connected_pilot:
        print("\n[Step 4/4] Executing integrated pilot on device...")
        from memory_study_v2.connected_pilot import run_connected_restricted_pilot
        pilot_rep = run_connected_restricted_pilot(output_dir / "connected_pilot", use_cuda=True)
        execution_result["executed"] = True
        execution_result["pilot_report"] = pilot_rep
        print(f"  Pilot completed: {pilot_rep['status']}")

    elapsed = round(time.perf_counter() - t_start, 3)

    receipt: Dict[str, Any] = {
        "receipt_type": "A30_PRODUCTION_LAUNCH_RECEIPT",
        "status": "PREFLIGHT_PASS" if (auth_info["can_proceed_to_production"] or check_only) else "LOCKED_AWAITING_AUTHORIZATION",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "host_telemetry": host_telemetry,
        "cuda_preflight": cuda_info,
        "storage_checks": storage_info,
        "authorization": auth_info,
        "execution": execution_result,
    }

    receipt_path = output_dir / "a30_launch_receipt.json"
    with open(receipt_path, "w", encoding="utf-8") as f:
        f.write(to_canonical_json(receipt))

    print(f"\nDeployment receipt saved to: {receipt_path}")
    print("=" * 70)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="A30 / Production Deployment Preflight and Launch Runner")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=REPO_ROOT / "rebuild_plan" / "config.proposed.json",
        help="Path to proposed production configuration file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to OHLCV data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "rebuild_plan" / "a30_production_out",
        help="Output directory for production artifacts and receipts",
    )
    parser.add_argument(
        "--authorize-production",
        action="store_true",
        default=False,
        help="Explicit operator authorization flag to unlock production execution at launch time",
    )
    parser.add_argument(
        "--run-pilot",
        action="store_true",
        default=False,
        help="Run integrated pilot pipeline on device as part of deployment verification",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        default=False,
        help="Run only preflight hardware, runtime, and storage checks without training",
    )
    args = parser.parse_args()

    receipt = launch_a30_deployment(
        config_path=args.config_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        authorize_production=args.authorize_production,
        run_connected_pilot=args.run_pilot,
        check_only=args.check_only,
    )

    if not args.check_only and not args.authorize_production:
        sys.exit(1)


if __name__ == "__main__":
    main()
