import json
import sys
from pathlib import Path

import pytest
import yaml

from src.orchestration.durable_runner import DurableExperimentRunner


def test_runner_skips_valid_outputs_and_rejects_source_drift(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("version-one", encoding="utf-8")
    output = tmp_path / "result.txt"
    state = tmp_path / "durable-state"
    command = f"from pathlib import Path; Path({str(output)!r}).write_text('ok', encoding='utf-8')"
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "id": "smoke",
                    "strict_environment": False,
                    "source_patterns": ["source.txt"],
                },
                "jobs": [
                    {
                        "id": "write-result",
                        "command": [sys.executable, "-c", command],
                        "expected_outputs": ["result.txt"],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    runner = DurableExperimentRunner(manifest_path, state, tmp_path)
    assert runner.run()["status"] == "completed"
    assert runner.plan()[0]["status"] == "completed"

    assert DurableExperimentRunner(manifest_path, state, tmp_path).run()["status"] == "completed"
    assert output.read_text(encoding="utf-8") == "ok"

    output.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="outputs changed"):
        DurableExperimentRunner(manifest_path, state, tmp_path).run()
    output.write_text("ok", encoding="utf-8")

    source.write_text("version-two", encoding="utf-8")
    with pytest.raises(RuntimeError, match="contract changed"):
        DurableExperimentRunner(manifest_path, state, tmp_path).run()


def test_source_migration_preserves_completed_jobs_and_updates_contract(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("version-one", encoding="utf-8")
    output = tmp_path / "result.txt"
    state = tmp_path / "durable-state"
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "id": "migration-smoke",
                    "strict_environment": False,
                    "source_patterns": ["source.txt"],
                },
                "jobs": [
                    {
                        "id": "write-result",
                        "command": [
                            sys.executable,
                            "-c",
                            f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
                        ],
                        "expected_outputs": ["result.txt"],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    assert DurableExperimentRunner(manifest_path, state, tmp_path).run()["status"] == "completed"

    source.write_text("version-two", encoding="utf-8")
    migrated = DurableExperimentRunner(manifest_path, state, tmp_path).migrate_source_contract(
        reason="Recover isolated AMP overflows without changing successful optimizer updates.",
        allowed_paths=["source.txt"],
        operator="test",
    )

    assert migrated["status"] == "migrated"
    assert migrated["completed_jobs_preserved"] == 1
    assert migrated["jobs_reset_to_pending"] == []
    record = Path(migrated["record"])
    assert record.is_file()
    assert DurableExperimentRunner(manifest_path, state, tmp_path).run()["status"] == "completed"


def test_source_migration_resets_only_failed_jobs(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("version-one", encoding="utf-8")
    state = tmp_path / "durable-state"
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "id": "failed-migration",
                    "strict_environment": False,
                    "source_patterns": ["source.txt"],
                },
                "jobs": [
                    {
                        "id": "fail",
                        "command": [sys.executable, "-c", "raise SystemExit(1)"],
                        "expected_outputs": ["missing.txt"],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="ended with status failed"):
        DurableExperimentRunner(manifest_path, state, tmp_path).run()

    source.write_text("version-two", encoding="utf-8")
    migrated = DurableExperimentRunner(manifest_path, state, tmp_path).migrate_source_contract(
        reason="Apply an operational recovery change while retaining the failed checkpoint.",
        allowed_paths=["source.txt"],
    )

    assert migrated["jobs_reset_to_pending"] == [
        {"job_id": "fail", "previous_status": "failed"}
    ]
    job_state = json.loads((state / "jobs" / "fail.json").read_text(encoding="utf-8"))
    assert job_state["status"] == "pending"
    assert job_state["status_before_migration"] == "failed"


def test_source_migration_rejects_undeclared_or_forbidden_changes(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    forbidden = tmp_path / "configs" / "training.yaml"
    forbidden.parent.mkdir()
    source.write_text("version-one", encoding="utf-8")
    forbidden.write_text("value: one", encoding="utf-8")
    manifest_path = tmp_path / "manifest.yaml"
    state = tmp_path / "durable-state"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "id": "migration-rejection",
                    "strict_environment": False,
                    "source_patterns": ["source.txt", "configs/**/*.yaml"],
                },
                "jobs": [
                    {
                        "id": "write",
                        "command": [
                            sys.executable,
                            "-c",
                            "from pathlib import Path; Path('result.txt').write_text('ok')",
                        ],
                        "expected_outputs": ["result.txt"],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    assert DurableExperimentRunner(manifest_path, state, tmp_path).run()["status"] == "completed"

    source.write_text("version-two", encoding="utf-8")
    with pytest.raises(RuntimeError, match="undeclared"):
        DurableExperimentRunner(manifest_path, state, tmp_path).migrate_source_contract(
            reason="Attempt a source migration without declaring the changed source file.",
            allowed_paths=["other.txt"],
        )

    source.write_text("version-one", encoding="utf-8")
    forbidden.write_text("value: two", encoding="utf-8")
    with pytest.raises(RuntimeError, match="requires a new run"):
        DurableExperimentRunner(manifest_path, state, tmp_path).migrate_source_contract(
            reason="Attempt a forbidden training configuration migration for validation.",
            allowed_paths=["configs/training.yaml"],
        )


def test_runtime_migration_accepts_only_declared_safe_host_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.txt"
    state = tmp_path / "durable-state"
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "run": {"id": "runtime-migration", "strict_environment": True},
                "jobs": [
                    {
                        "id": "write",
                        "command": [
                            sys.executable,
                            "-c",
                            f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
                        ],
                        "expected_outputs": ["result.txt"],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    assert DurableExperimentRunner(manifest_path, state, tmp_path).run()["status"] == "completed"
    original = json.loads((state / "contract.json").read_text(encoding="utf-8"))["runtime"]

    monkeypatch.setattr(
        "src.orchestration.durable_runner._runtime_fingerprint",
        lambda: {**original, "release": "replacement-kernel"},
    )
    migrated = DurableExperimentRunner(manifest_path, state, tmp_path).migrate_runtime_contract(
        reason="Accept a spot-instance kernel release change with an identical ML runtime.",
        allowed_fields=["release"],
        operator="test",
    )
    assert migrated["changed_fields"] == ["release"]
    assert migrated["completed_jobs_preserved"] == 1
    record = json.loads(Path(migrated["record"]).read_text(encoding="utf-8"))
    assert record["changes"]["release"] == {
        "before": original["release"],
        "after": "replacement-kernel",
    }


def test_runtime_migration_rejects_ml_runtime_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "durable-state"
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "run": {"id": "unsafe-runtime", "strict_environment": True},
                "jobs": [
                    {
                        "id": "write",
                        "command": [sys.executable, "-c", "from pathlib import Path; Path('x').write_text('x')"],
                        "expected_outputs": ["x"],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    assert DurableExperimentRunner(manifest_path, state, tmp_path).run()["status"] == "completed"
    original = json.loads((state / "contract.json").read_text(encoding="utf-8"))["runtime"]
    monkeypatch.setattr(
        "src.orchestration.durable_runner._runtime_fingerprint",
        lambda: {**original, "torch": "different"},
    )
    with pytest.raises(RuntimeError, match="unsafe fields"):
        DurableExperimentRunner(manifest_path, state, tmp_path).migrate_runtime_contract(
            reason="Attempt to accept an incompatible machine-learning runtime change.",
            allowed_fields=["torch"],
        )


def test_source_migration_can_atomically_accept_safe_host_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "runner.py"
    source.write_text("old", encoding="utf-8")
    output = tmp_path / "result.txt"
    state = tmp_path / "durable-state"
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "id": "combined-migration",
                    "strict_environment": True,
                    "source_patterns": ["runner.py"],
                },
                "jobs": [
                    {
                        "id": "write",
                        "command": [
                            sys.executable,
                            "-c",
                            f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
                        ],
                        "expected_outputs": ["result.txt"],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    assert DurableExperimentRunner(manifest_path, state, tmp_path).run()["status"] == "completed"
    original = json.loads((state / "contract.json").read_text(encoding="utf-8"))["runtime"]
    source.write_text("new", encoding="utf-8")
    monkeypatch.setattr(
        "src.orchestration.durable_runner._runtime_fingerprint",
        lambda: {**original, "release": "replacement-kernel"},
    )

    migrated = DurableExperimentRunner(manifest_path, state, tmp_path).migrate_source_contract(
        reason="Accept operational runner recovery and spot-instance kernel replacement together.",
        allowed_paths=["runner.py"],
        allowed_runtime_fields=["release"],
        operator="test",
    )

    record = json.loads(Path(migrated["record"]).read_text(encoding="utf-8"))
    assert record["runtime_changes"]["release"]["after"] == "replacement-kernel"
