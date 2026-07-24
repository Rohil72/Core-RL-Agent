"""Manifest-driven, resumable execution for long research experiments."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import fnmatch
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


FORBIDDEN_SOURCE_MIGRATION_PATTERNS = (
    "configs/**",
    "requirements*.txt",
    "src/data/**",
    "src/features/**",
    "src/losses/**",
    "src/models/**",
    "scripts/build_final_testbed.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        absolute_pattern = str((root / pattern).resolve()) if not Path(pattern).is_absolute() else pattern
        for match in glob.glob(absolute_pattern, recursive=True):
            path = Path(match)
            if path.is_file():
                files.add(path.resolve())
            elif path.is_dir():
                files.update(item.resolve() for item in path.rglob("*") if item.is_file())
    return sorted(files, key=lambda item: str(item).lower())


def _fingerprint_files(root: Path, patterns: Iterable[str]) -> dict[str, Any]:
    rows = []
    for path in _resolved_files(root, patterns):
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = str(path)
        rows.append({"path": label, "size": path.stat().st_size, "sha256": _file_sha256(path)})
    digest = _sha256_bytes(json.dumps(rows, sort_keys=True).encode("utf-8"))
    return {"sha256": digest, "file_count": len(rows), "files": rows}


def _fingerprint_changes(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return file-level additions, removals, and modifications."""
    previous_files = {row["path"]: row for row in previous.get("files", [])}
    current_files = {row["path"]: row for row in current.get("files", [])}
    changes: list[dict[str, Any]] = []
    for path in sorted(set(previous_files) | set(current_files)):
        before = previous_files.get(path)
        after = current_files.get(path)
        if before == after:
            continue
        changes.append(
            {
                "path": path.replace("\\", "/"),
                "change": "added" if before is None else "removed" if after is None else "modified",
                "before": before,
                "after": after,
            }
        )
    return changes


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    normalized = path.replace("\\", "/")
    return any(fnmatch.fnmatchcase(normalized, pattern.replace("\\", "/")) for pattern in patterns)


def _runtime_fingerprint() -> dict[str, Any]:
    result: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import torch

        result.update(
            {
                "torch": str(torch.__version__),
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "cuda_available": torch.cuda.is_available(),
            }
        )
        if torch.cuda.is_available():
            devices = []
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "name": properties.name,
                        "compute_capability": [properties.major, properties.minor],
                        "total_memory": int(properties.total_memory),
                    }
                )
            result["cuda_devices"] = devices
    except ImportError:
        result["torch"] = None
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        result["nvidia_driver"] = sorted(set(driver.stdout.split()))
    except (FileNotFoundError, subprocess.SubprocessError):
        result["nvidia_driver"] = None
    return result


def _python_runtime_fingerprint(executable: str, cwd: Path) -> dict[str, Any] | None:
    """Inspect Python job environments, including isolated RL environments."""
    if "python" not in Path(executable).name.lower():
        return None
    probe = (
        "import json,platform\n"
        "result={'python':platform.python_version()}\n"
        "try:\n"
        " import torch\n"
        " result.update(torch=str(torch.__version__),cuda=torch.version.cuda,"
        "cudnn=torch.backends.cudnn.version(),cuda_available=torch.cuda.is_available(),"
        "devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\n"
        "except ImportError:\n"
        " result['torch']=None\n"
        "print(json.dumps(result,sort_keys=True))\n"
    )
    try:
        completed = subprocess.run(
            [executable, "-c", probe],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {"probe_error": str(exc)}


@dataclass(frozen=True)
class JobSpec:
    """One restartable command in the experiment DAG."""

    job_id: str
    command: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    cwd: str = "."
    stage: str = "default"
    uses_gpu: bool = False
    gpu_count: int = 1
    estimated_hours: float = 0.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JobSpec":
        command = value.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
            raise ValueError(f"Job {value.get('id')} command must be a non-empty string list.")
        return cls(
            job_id=str(value["id"]),
            command=tuple(command),
            depends_on=tuple(map(str, value.get("depends_on", []))),
            expected_outputs=tuple(map(str, value.get("expected_outputs", []))),
            inputs=tuple(map(str, value.get("inputs", []))),
            environment={str(key): str(item) for key, item in value.get("environment", {}).items()},
            cwd=str(value.get("cwd", ".")),
            stage=str(value.get("stage", "default")),
            uses_gpu=bool(value.get("uses_gpu", False)),
            gpu_count=max(int(value.get("gpu_count", 1)), 1),
            estimated_hours=max(float(value.get("estimated_hours", 0.0)), 0.0),
        )


@dataclass(frozen=True)
class ExperimentManifest:
    """Immutable experiment contract loaded from YAML or JSON."""

    run_id: str
    jobs: tuple[JobSpec, ...]
    input_patterns: tuple[str, ...] = ()
    source_patterns: tuple[str, ...] = (
        "src/**/*.py",
        "scripts/**/*.py",
        "configs/**/*.yaml",
        "requirements*.txt",
    )
    strict_environment: bool = True
    stale_lock_seconds: int = 900
    hardware: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "ExperimentManifest":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Experiment manifest must be a mapping.")
        run = raw.get("run", {})
        jobs = tuple(JobSpec.from_dict(value) for value in raw.get("jobs", []))
        manifest = cls(
            run_id=str(run.get("id") or path.stem),
            jobs=jobs,
            input_patterns=tuple(map(str, run.get("inputs", []))),
            source_patterns=tuple(map(str, run.get("source_patterns", cls.source_patterns))),
            strict_environment=bool(run.get("strict_environment", True)),
            stale_lock_seconds=int(run.get("stale_lock_seconds", 900)),
            hardware=dict(run.get("hardware", {})),
            budgets={str(key): dict(value) for key, value in run.get("budgets", {}).items()},
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if not self.jobs:
            raise ValueError("Experiment manifest has no jobs.")
        ids = [job.job_id for job in self.jobs]
        if len(ids) != len(set(ids)):
            raise ValueError("Experiment job IDs must be unique.")
        known = set(ids)
        for job in self.jobs:
            missing = set(job.depends_on) - known
            if missing:
                raise ValueError(f"Job {job.job_id} has unknown dependencies: {sorted(missing)}")
            if not job.expected_outputs:
                raise ValueError(f"Job {job.job_id} must declare at least one expected output.")
        pending = set(ids)
        completed: set[str] = set()
        while pending:
            ready = {job.job_id for job in self.jobs if job.job_id in pending and set(job.depends_on) <= completed}
            if not ready:
                raise ValueError("Experiment dependencies contain a cycle.")
            pending -= ready
            completed |= ready


class DurableExperimentRunner:
    """Execute an immutable experiment DAG with restart-safe state and logs."""

    def __init__(
        self,
        manifest_path: Path,
        state_dir: Path | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.root = (project_root or Path.cwd()).resolve()
        self.manifest = ExperimentManifest.load(self.manifest_path)
        self.state_dir = (state_dir or self.root / ".experiment_state" / self.manifest.run_id).resolve()
        self.jobs_dir = self.state_dir / "jobs"
        self.logs_dir = self.state_dir / "logs"
        self.lock_path = self.state_dir / "runner.lock"
        self.contract_path = self.state_dir / "contract.json"
        self.stop_requested = False
        self.active_process: subprocess.Popen[str] | None = None
        self.runtime_fingerprint: dict[str, Any] | None = None

    def _manifest_digest(self) -> str:
        return _sha256_bytes(self.manifest_path.read_bytes())

    def _build_contract(self) -> dict[str, Any]:
        job_runtimes = {}
        runtime_cache: dict[tuple[str, str], dict[str, Any] | None] = {}
        for job in self.manifest.jobs:
            cwd = (self.root / job.cwd).resolve() if not Path(job.cwd).is_absolute() else Path(job.cwd)
            cache_key = (job.command[0], str(cwd))
            if cache_key not in runtime_cache:
                runtime_cache[cache_key] = _python_runtime_fingerprint(job.command[0], cwd)
            runtime = runtime_cache[cache_key]
            if runtime is not None:
                job_runtimes[job.job_id] = runtime
        return {
            "format_version": 1,
            "run_id": self.manifest.run_id,
            "manifest_sha256": self._manifest_digest(),
            "source": _fingerprint_files(self.root, self.manifest.source_patterns),
            "inputs": _fingerprint_files(self.root, self.manifest.input_patterns),
            "runtime": _runtime_fingerprint(),
            "job_runtimes": job_runtimes,
            "hardware_allowance": self.manifest.hardware,
            "budgets": self.manifest.budgets,
        }

    def _validate_contract(self) -> dict[str, Any]:
        current = self._build_contract()
        self.runtime_fingerprint = current["runtime"]
        required_runtime = self.manifest.hardware.get("required_runtime_fingerprint")
        if required_runtime is not None and current["runtime"] != required_runtime:
            raise RuntimeError(
                "Current runtime does not match the development testbed pinned for confirmation."
            )
        required_source = self.manifest.hardware.get("required_source_fingerprint")
        if required_source is not None and current["source"] != required_source:
            raise RuntimeError(
                "Current source does not match the development testbed pinned for confirmation."
            )
        if not self.contract_path.exists():
            current["created_at"] = _utc_now()
            _atomic_json(self.contract_path, current)
            return current
        saved = json.loads(self.contract_path.read_text(encoding="utf-8"))
        immutable_keys = ["run_id", "manifest_sha256", "source", "inputs", "job_runtimes"]
        if self.manifest.strict_environment:
            immutable_keys.append("runtime")
        changed = [key for key in immutable_keys if saved.get(key) != current.get(key)]
        if changed:
            raise RuntimeError(
                "Experiment contract changed in: " + ", ".join(changed) + ". "
                "Use a new run ID/state directory instead of mixing testbeds or artifacts."
            )
        return saved

    def migrate_source_contract(
        self,
        *,
        reason: str,
        allowed_paths: Iterable[str],
        operator: str | None = None,
    ) -> dict[str, Any]:
        """Audit and accept an operational source-only change for checkpoint resume."""
        reason = reason.strip()
        if len(reason) < 20:
            raise ValueError("Migration reason must contain at least 20 characters.")
        normalized_allowed = tuple(
            sorted({str(path).replace("\\", "/").strip() for path in allowed_paths if str(path).strip()})
        )
        if not normalized_allowed:
            raise ValueError("At least one explicitly allowed source path is required.")
        if any(Path(path).is_absolute() or ".." in Path(path).parts for path in normalized_allowed):
            raise ValueError("Allowed source paths must be project-relative and cannot contain '..'.")

        self._acquire_lock()
        try:
            if not self.contract_path.exists():
                raise RuntimeError("Cannot migrate a run before its initial contract has been created.")
            saved = json.loads(self.contract_path.read_text(encoding="utf-8"))
            current = self._build_contract()
            self.runtime_fingerprint = current["runtime"]

            immutable_keys = ["run_id", "manifest_sha256", "inputs", "job_runtimes"]
            if self.manifest.strict_environment:
                immutable_keys.append("runtime")
            incompatible = [key for key in immutable_keys if saved.get(key) != current.get(key)]
            if incompatible:
                raise RuntimeError(
                    "Source migration refused because non-source contract fields changed: "
                    + ", ".join(incompatible)
                )
            required_source = self.manifest.hardware.get("required_source_fingerprint")
            if required_source is not None:
                raise RuntimeError(
                    "Source migration is forbidden for a confirmation run with a pinned source fingerprint."
                )

            changes = _fingerprint_changes(saved.get("source", {}), current["source"])
            if not changes:
                raise RuntimeError("Source migration requested, but the source fingerprint is unchanged.")
            undeclared = [
                change["path"]
                for change in changes
                if not _matches_any(change["path"], normalized_allowed)
            ]
            if undeclared:
                raise RuntimeError(
                    "Source migration contains undeclared changed files: " + ", ".join(undeclared)
                )
            forbidden = [
                change["path"]
                for change in changes
                if _matches_any(change["path"], FORBIDDEN_SOURCE_MIGRATION_PATTERNS)
            ]
            if forbidden:
                raise RuntimeError(
                    "Source migration touches architecture, data, loss, configuration, dependency, "
                    "or testbed-design files and requires a new run: " + ", ".join(forbidden)
                )

            migration_dir = self.state_dir / "migrations"
            migration_dir.mkdir(parents=True, exist_ok=True)
            existing_records = sorted(migration_dir.glob("*.json"))
            sequence = len(existing_records) + 1
            previous_record_sha256 = (
                _file_sha256(existing_records[-1]) if existing_records else None
            )
            reset_jobs: list[dict[str, str]] = []
            completed_jobs: list[str] = []
            for job in self.manifest.jobs:
                state = self._job_state(job.job_id)
                if not state:
                    continue
                status = str(state.get("status", ""))
                if status == "completed":
                    self._is_complete(job)
                    completed_jobs.append(job.job_id)
                elif status in {"failed", "interrupted"}:
                    reset_jobs.append({"job_id": job.job_id, "previous_status": status})

            record = {
                "format_version": 1,
                "sequence": sequence,
                "run_id": self.manifest.run_id,
                "migrated_at": _utc_now(),
                "operator": operator or os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
                "reason": reason,
                "previous_record_sha256": previous_record_sha256,
                "old_source_sha256": saved.get("source", {}).get("sha256"),
                "new_source_sha256": current["source"]["sha256"],
                "allowed_paths": list(normalized_allowed),
                "changes": changes,
                "verified_unchanged": immutable_keys,
                "checkpoint_resume_contract": [
                    "training_config_fingerprint",
                    "runtime_fingerprint",
                    "model_state",
                    "optimizer_state",
                    "grad_scaler_state",
                    "rng_state",
                    "data_loader_position",
                ],
                "completed_jobs_preserved": completed_jobs,
                "jobs_reset_to_pending": reset_jobs,
            }
            record_payload_sha256 = _sha256_bytes(
                json.dumps(record, sort_keys=True).encode("utf-8")
            )
            record["record_payload_sha256"] = record_payload_sha256
            record_path = migration_dir / f"{sequence:04d}_source_migration.json"
            _atomic_json(record_path, record)
            record_file_sha256 = _file_sha256(record_path)

            for reset in reset_jobs:
                state = self._job_state(reset["job_id"])
                assert state is not None
                state["status"] = "pending"
                state["migration_sequence"] = sequence
                state["status_before_migration"] = reset["previous_status"]
                state["migrated_at"] = record["migrated_at"]
                _atomic_json(self._job_path(reset["job_id"]), state)

            migrated_contract = current
            migrated_contract["created_at"] = saved.get("created_at", record["migrated_at"])
            migrated_contract["last_migrated_at"] = record["migrated_at"]
            migrated_contract["migration_count"] = sequence
            migrated_contract["migration_head_sha256"] = record_file_sha256
            _atomic_json(self.contract_path, migrated_contract)
            return {
                "status": "migrated",
                "run_id": self.manifest.run_id,
                "record": str(record_path),
                "old_source_sha256": record["old_source_sha256"],
                "new_source_sha256": record["new_source_sha256"],
                "changed_paths": [change["path"] for change in changes],
                "completed_jobs_preserved": len(completed_jobs),
                "jobs_reset_to_pending": reset_jobs,
            }
        finally:
            self._release_lock()

    def _validate_hardware(self, job: JobSpec) -> None:
        if not job.uses_gpu:
            return
        runtime = self.runtime_fingerprint or _runtime_fingerprint()
        allowance = self.manifest.hardware
        if not runtime.get("cuda_available"):
            raise RuntimeError(f"GPU job {job.job_id} requires CUDA, but CUDA is unavailable.")
        devices = runtime.get("cuda_devices", [])
        if len(devices) < job.gpu_count:
            raise RuntimeError(
                f"GPU job {job.job_id} requires {job.gpu_count} GPU(s), found {len(devices)}."
            )
        minimum_gb = float(allowance.get("minimum_vram_gb", 0.0))
        selected = devices[: job.gpu_count]
        if any(float(device["total_memory"]) / 1024**3 < minimum_gb for device in selected):
            raise RuntimeError(
                f"GPU job {job.job_id} requires at least {minimum_gb:g} GB VRAM per GPU."
            )
        minimum_capability = tuple(allowance.get("minimum_compute_capability", [0, 0]))
        if any(tuple(device["compute_capability"]) < minimum_capability for device in selected):
            raise RuntimeError(
                f"GPU job {job.job_id} requires compute capability {minimum_capability} or newer."
            )

    def _validate_storage(self) -> None:
        minimum_gb = float(self.manifest.hardware.get("minimum_free_storage_gb", 0.0))
        if minimum_gb <= 0:
            return
        free_gb = shutil.disk_usage(self.state_dir).free / 1024**3
        if free_gb < minimum_gb:
            raise RuntimeError(
                f"Experiment storage has {free_gb:.1f} GB free; {minimum_gb:.1f} GB is required."
            )

    def _used_gpu_hours(self, stage: str) -> float:
        total = 0.0
        for job in self.manifest.jobs:
            if job.stage != stage or not job.uses_gpu:
                continue
            state = self._job_state(job.job_id) or {}
            total += float(state.get("elapsed_seconds", 0.0)) / 3600.0 * job.gpu_count
        return total

    def _validate_budget(self, job: JobSpec) -> None:
        if not job.uses_gpu:
            return
        budget = self.manifest.budgets.get(job.stage, {})
        if not budget:
            return
        used = self._used_gpu_hours(job.stage)
        projected = used + job.estimated_hours * job.gpu_count
        maximum_hours = budget.get("max_gpu_hours")
        if maximum_hours is not None and projected > float(maximum_hours):
            raise RuntimeError(
                f"Starting {job.job_id} would project {projected:.2f} GPU-hours in stage "
                f"{job.stage}, above its {float(maximum_hours):.2f}-hour budget."
            )
        hourly_cost = budget.get("hourly_cost_inr")
        maximum_cost = budget.get("max_cost_inr")
        if hourly_cost is not None and maximum_cost is not None:
            projected_cost = projected * float(hourly_cost)
            if projected_cost > float(maximum_cost):
                raise RuntimeError(
                    f"Starting {job.job_id} would project INR {projected_cost:.2f}, "
                    f"above the INR {float(maximum_cost):.2f} budget for {job.stage}."
                )

    def _lock_payload(self) -> dict[str, Any]:
        return {"pid": os.getpid(), "host": socket.gethostname(), "heartbeat": _utc_now()}

    def _pid_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _acquire_lock(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            existing = json.loads(self.lock_path.read_text(encoding="utf-8"))
            same_host_alive = existing.get("host") == socket.gethostname() and self._pid_alive(int(existing.get("pid", -1)))
            heartbeat = datetime.fromisoformat(existing["heartbeat"])
            stale = (datetime.now(timezone.utc) - heartbeat).total_seconds() > self.manifest.stale_lock_seconds
            if same_host_alive or not stale:
                raise RuntimeError(f"Experiment runner is already active: {existing}")
            self.lock_path.unlink()
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(self._lock_payload(), handle, indent=2)

    def _heartbeat(self) -> None:
        _atomic_json(self.lock_path, self._lock_payload())

    def _heartbeat_loop(self, stop: threading.Event) -> None:
        while not stop.wait(30):
            try:
                self._heartbeat()
            except OSError:
                return

    def _release_lock(self) -> None:
        if self.lock_path.exists():
            self.lock_path.unlink()

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _job_state(self, job_id: str) -> dict[str, Any] | None:
        path = self._job_path(job_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def _output_exists(self, value: str) -> bool:
        path = (self.root / value).resolve() if not Path(value).is_absolute() else Path(value)
        return path.is_file() and path.stat().st_size > 0 or path.is_dir() and any(path.iterdir())

    def _is_complete(self, job: JobSpec) -> bool:
        state = self._job_state(job.job_id)
        if not state or state.get("status") != "completed":
            return False
        if not all(self._output_exists(output) for output in job.expected_outputs):
            raise RuntimeError(f"Completed job {job.job_id} is missing a declared output.")
        current = _fingerprint_files(self.root, job.expected_outputs)
        if state.get("outputs") != current:
            raise RuntimeError(f"Completed outputs changed for job {job.job_id}; refusing silent reuse.")
        current_inputs = _fingerprint_files(self.root, job.inputs)
        if state.get("inputs") != current_inputs:
            raise RuntimeError(f"Inputs changed for completed job {job.job_id}; use a new run ID.")
        return True

    def _ordered_jobs(self) -> list[JobSpec]:
        remaining = {job.job_id: job for job in self.manifest.jobs}
        ordered: list[JobSpec] = []
        emitted: set[str] = set()
        while remaining:
            ready = [job for job in self.manifest.jobs if job.job_id in remaining and set(job.depends_on) <= emitted]
            for job in ready:
                ordered.append(job)
                emitted.add(job.job_id)
                remaining.pop(job.job_id)
        return ordered

    def plan(self, selected_stages: set[str] | None = None) -> list[dict[str, Any]]:
        """Return the ordered execution plan without changing state."""
        return [
            {
                "id": job.job_id,
                "status": "completed" if self._is_complete(job) else "pending",
                "depends_on": list(job.depends_on),
                "command": list(job.command),
                "stage": job.stage,
                "uses_gpu": job.uses_gpu,
                "estimated_hours": job.estimated_hours,
            }
            for job in self._ordered_jobs()
            if selected_stages is None or job.stage in selected_stages
        ]

    def _signal_handler(self, signum: int, _frame: Any) -> None:
        self.stop_requested = True
        if self.active_process is not None and self.active_process.poll() is None:
            try:
                if os.name == "nt":
                    self.active_process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.active_process.send_signal(signal.SIGTERM)
            except OSError:
                pass

    def _run_job(self, job: JobSpec) -> None:
        previous = self._job_state(job.job_id) or {}
        missing_inputs = [pattern for pattern in job.inputs if not _resolved_files(self.root, [pattern])]
        if missing_inputs:
            raise FileNotFoundError(f"Job {job.job_id} has missing inputs: {missing_inputs}")
        inputs = _fingerprint_files(self.root, job.inputs)
        if previous.get("inputs") is not None and previous["inputs"] != inputs:
            raise RuntimeError(f"Inputs changed between attempts for job {job.job_id}; use a new run ID.")
        attempt = int(previous.get("attempt", 0)) + 1
        started = time.monotonic()
        state = {
            "job_id": job.job_id,
            "status": "running",
            "attempt": attempt,
            "started_at": _utc_now(),
            "command": list(job.command),
            "stage": job.stage,
            "inputs": inputs,
        }
        _atomic_json(self._job_path(job.job_id), state)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"{job.job_id}.attempt_{attempt}.log"
        environment = os.environ.copy()
        environment.update(job.environment)
        if job.uses_gpu:
            allowance = self.manifest.hardware
            environment.setdefault(
                "CORE_RL_VRAM_FRACTION",
                str(allowance.get("vram_fraction", 0.80)),
            )
            environment.setdefault(
                "CUDA_VISIBLE_DEVICES",
                str(allowance.get("cuda_visible_devices", "0")),
            )
        cwd = (self.root / job.cwd).resolve() if not Path(job.cwd).is_absolute() else Path(job.cwd)
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"[{_utc_now()}] command={list(job.command)!r}\n")
            self.active_process = subprocess.Popen(
                list(job.command),
                cwd=cwd,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=creationflags,
            )
            assert self.active_process.stdout is not None
            for line in self.active_process.stdout:
                print(line, end="")
                log.write(line)
            return_code = self.active_process.wait()
        self.active_process = None
        state.update(
            {
                "status": "interrupted" if self.stop_requested else "completed" if return_code == 0 else "failed",
                "finished_at": _utc_now(),
                "return_code": return_code,
                "log": str(log_path),
                "elapsed_seconds": float(previous.get("elapsed_seconds", 0.0)) + (time.monotonic() - started),
            }
        )
        if return_code == 0 and not all(self._output_exists(output) for output in job.expected_outputs):
            state["status"] = "failed"
            state["error"] = "Command succeeded but one or more expected outputs are absent or empty."
        if state["status"] == "completed":
            state["outputs"] = _fingerprint_files(self.root, job.expected_outputs)
        _atomic_json(self._job_path(job.job_id), state)
        if state["status"] != "completed":
            raise RuntimeError(f"Job {job.job_id} ended with status {state['status']}; see {log_path}")

    def run(
        self,
        selected_jobs: set[str] | None = None,
        selected_stages: set[str] | None = None,
    ) -> dict[str, Any]:
        """Run pending jobs and return a concise durable-state summary."""
        self._acquire_lock()
        previous_handlers: dict[int, Any] = {}
        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(heartbeat_stop,),
            name="experiment-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            self._validate_contract()
            self._validate_storage()
            termination_signals = {
                signal.SIGINT,
                getattr(signal, "SIGTERM", signal.SIGINT),
                getattr(signal, "SIGBREAK", signal.SIGINT),
            }
            for signum in termination_signals:
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._signal_handler)
            for job in self._ordered_jobs():
                if selected_jobs is not None and job.job_id not in selected_jobs:
                    continue
                if selected_stages is not None and job.stage not in selected_stages:
                    continue
                if self._is_complete(job):
                    print(f"SKIP {job.job_id}: validated completion state")
                    continue
                incomplete_dependencies = [dependency for dependency in job.depends_on if not self._is_complete(next(item for item in self.manifest.jobs if item.job_id == dependency))]
                if incomplete_dependencies:
                    raise RuntimeError(f"Job {job.job_id} has incomplete dependencies: {incomplete_dependencies}")
                self._validate_hardware(job)
                self._validate_budget(job)
                print(f"RUN  {job.job_id}: {' '.join(job.command)}")
                self._run_job(job)
                if self.stop_requested:
                    break
            plan = self.plan()
            selected_plan = self.plan(selected_stages)
            summary = {
                "run_id": self.manifest.run_id,
                "state_dir": str(self.state_dir),
                "completed": sum(item["status"] == "completed" for item in plan),
                "total": len(plan),
                "status": "completed" if all(item["status"] == "completed" for item in plan) else "incomplete",
                "selected_completed": sum(item["status"] == "completed" for item in selected_plan),
                "selected_total": len(selected_plan),
                "gpu_hours_by_stage": {
                    stage: self._used_gpu_hours(stage)
                    for stage in sorted({job.stage for job in self.manifest.jobs if job.uses_gpu})
                },
            }
            _atomic_json(self.state_dir / "summary.json", summary)
            return summary
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5)
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            self._release_lock()
