"""Acceptance Test A33: Resource monitoring, thresholds, and stress fixtures.

Acceptance criteria:
- CPU RAM, GPU allocation and disk free thresholds pass stress fixtures.
- Free disk < 20 GB triggers safe pause error.
- Exceeding host RAM or GPU dedicated memory triggers error.
"""

from pathlib import Path
import pytest

from memory_study_v2.resources import (
    LAPTOP_CEILINGS,
    VM_CEILINGS,
    LowDiskPauseError,
    ResourceCeilings,
    ResourceExceededError,
    ResourceSnapshot,
    check_resource_thresholds,
    get_current_resource_snapshot,
)


def test_live_resource_snapshot():
    snapshot = get_current_resource_snapshot()
    assert snapshot.process_rss_gb > 0
    assert snapshot.free_disk_gb > 0
    assert snapshot.status == "PASS"


def test_resource_thresholds_nominal_pass():
    # Nominal laptop usage
    snap = ResourceSnapshot(
        host_ram_used_gb=8.0,
        process_rss_gb=2.5,
        gpu_allocated_gb=1.2,
        free_disk_gb=45.0,
    )
    assert check_resource_thresholds(snap, LAPTOP_CEILINGS) is True
    assert snap.status == "PASS"

    # Nominal VM usage
    snap_vm = ResourceSnapshot(
        host_ram_used_gb=32.0,
        process_rss_gb=16.0,
        gpu_allocated_gb=12.0,
        free_disk_gb=75.0,
    )
    assert check_resource_thresholds(snap_vm, VM_CEILINGS) is True
    assert snap_vm.status == "PASS"


def test_low_disk_safety_pause():
    # Free disk falls to 18 GB (< 20 GB ceiling)
    snap = ResourceSnapshot(
        host_ram_used_gb=8.0,
        process_rss_gb=2.0,
        gpu_allocated_gb=1.0,
        free_disk_gb=18.5,
    )
    with pytest.raises(LowDiskPauseError, match="below minimum safety threshold 20.00 GB"):
        check_resource_thresholds(snap, LAPTOP_CEILINGS, strict=True)

    # Non-strict returns False and populates alert message
    assert check_resource_thresholds(snap, LAPTOP_CEILINGS, strict=False) is False
    assert snap.status == "FAIL_LOW_DISK"
    assert "pause safely without deleting unexported evidence" in snap.alert_message


def test_memory_ceilings_exceeded():
    # Laptop RAM ceiling exceeded (e.g. 12 GB > 10 GB)
    snap_high_ram = ResourceSnapshot(
        host_ram_used_gb=14.0,
        process_rss_gb=11.5,
        gpu_allocated_gb=1.0,
        free_disk_gb=50.0,
    )
    with pytest.raises(ResourceExceededError, match="Process RSS 11.50 GB exceeds ceiling 10.00 GB"):
        check_resource_thresholds(snap_high_ram, LAPTOP_CEILINGS, strict=True)

    # Laptop GPU ceiling exceeded (e.g. 3.5 GB > 3.2 GB)
    snap_high_gpu = ResourceSnapshot(
        host_ram_used_gb=8.0,
        process_rss_gb=2.0,
        gpu_allocated_gb=3.6,
        free_disk_gb=50.0,
    )
    with pytest.raises(ResourceExceededError, match="GPU allocated memory 3.60 GB exceeds ceiling 3.20 GB"):
        check_resource_thresholds(snap_high_gpu, LAPTOP_CEILINGS, strict=True)
