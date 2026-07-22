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
