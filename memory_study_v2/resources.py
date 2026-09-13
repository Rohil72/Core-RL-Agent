"""Resource monitoring, threshold enforcement, and stress fixtures (v2).

Acceptance criteria addressed:
- A33: CPU RAM, GPU allocation and disk free thresholds pass stress fixtures.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import psutil
import torch


class LowDiskPauseError(Exception):
    """Raised when available storage falls below the 20 GB safety ceiling."""
    pass


class ResourceExceededError(Exception):
    """Raised when RAM or GPU usage exceeds operating limits."""
    pass


@dataclass
class ResourceCeilings:
    max_host_ram_gb: float
    max_gpu_allocated_gb: float
    min_free_disk_gb: float = 20.0
    max_working_artifacts_gb: float = 60.0


# Operational ceilings defined in REBUILD_IMPLEMENTATION_PLAN.md Section 12.3:
# Laptop ceilings: 10GB application RAM and 3.2GB GPU allocation on its 4GB dedicated device.
LAPTOP_CEILINGS = ResourceCeilings(
    max_host_ram_gb=10.0,
    max_gpu_allocated_gb=3.2,
    min_free_disk_gb=20.0,
    max_working_artifacts_gb=40.0,
)

# VM ceilings (A30 target): 48GB total application RAM, 20GB GPU allocation, 60GB working artifacts, >= 20GB free disk.
VM_CEILINGS = ResourceCeilings(
    max_host_ram_gb=48.0,
    max_gpu_allocated_gb=20.0,
    min_free_disk_gb=20.0,
    max_working_artifacts_gb=60.0,
)


@dataclass
class ResourceSnapshot:
    host_ram_used_gb: float
    process_rss_gb: float
    gpu_allocated_gb: float
    free_disk_gb: float
    status: str = "PASS"
    alert_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_current_resource_snapshot(disk_path: Union[str, Path] = ".") -> ResourceSnapshot:
    """Query live system resources on current machine."""
    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / (1024 ** 3)
    vm = psutil.virtual_memory()
    host_ram_used_gb = vm.used / (1024 ** 3)

    gpu_allocated_gb = 0.0
    if torch.cuda.is_available():
        gpu_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)

    disk_stat = shutil.disk_usage(str(disk_path))
    free_disk_gb = disk_stat.free / (1024 ** 3)

    return ResourceSnapshot(
        host_ram_used_gb=round(host_ram_used_gb, 3),
        process_rss_gb=round(rss_gb, 3),
        gpu_allocated_gb=round(gpu_allocated_gb, 3),
        free_disk_gb=round(free_disk_gb, 3),
        status="PASS",
    )


def check_resource_thresholds(
    snapshot: ResourceSnapshot,
    ceilings: ResourceCeilings,
    strict: bool = True,
) -> bool:
    """Validate resource snapshot against ceilings.
    
    Raises:
        LowDiskPauseError: if free disk < min_free_disk_gb (default 20.0 GB).
        ResourceExceededError: if RAM or GPU allocation exceeds configured ceilings.
    """
    if snapshot.free_disk_gb < ceilings.min_free_disk_gb:
        msg = (
            f"Free disk {snapshot.free_disk_gb:.2f} GB is below minimum safety threshold "
            f"{ceilings.min_free_disk_gb:.2f} GB. Execution must pause safely without deleting unexported evidence."
        )
        snapshot.status = "FAIL_LOW_DISK"
        snapshot.alert_message = msg
        if strict:
            raise LowDiskPauseError(msg)
        return False

    if snapshot.process_rss_gb > ceilings.max_host_ram_gb:
        msg = (
            f"Process RSS {snapshot.process_rss_gb:.2f} GB exceeds ceiling "
            f"{ceilings.max_host_ram_gb:.2f} GB."
        )
        snapshot.status = "FAIL_RAM_CEILING"
        snapshot.alert_message = msg
        if strict:
            raise ResourceExceededError(msg)
        return False

    if snapshot.gpu_allocated_gb > ceilings.max_gpu_allocated_gb:
        msg = (
            f"GPU allocated memory {snapshot.gpu_allocated_gb:.2f} GB exceeds ceiling "
            f"{ceilings.max_gpu_allocated_gb:.2f} GB."
        )
        snapshot.status = "FAIL_GPU_CEILING"
        snapshot.alert_message = msg
        if strict:
            raise ResourceExceededError(msg)
        return False

    snapshot.status = "PASS"
    snapshot.alert_message = None
    return True
