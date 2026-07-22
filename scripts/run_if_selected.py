"""Execute one manifest command only when its policy passed the pilot gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--status-output", required=True)
    parser.add_argument("--artifact")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    selected = args.algorithm in selection.get("selected_algorithms", [])
    payload = {
        "algorithm": args.algorithm,
        "selected": selected,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command,
    }
    if selected:
        if not command:
            raise ValueError("Selected policy wrapper requires a command after --.")
        subprocess.run(command, check=True)
        if args.artifact and not Path(args.artifact).exists():
            raise FileNotFoundError(f"Selected command did not produce {args.artifact}")
        payload["status"] = "completed"
        payload["artifact"] = args.artifact
    else:
        payload["status"] = "skipped_not_selected"
    destination = Path(args.status_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
