"""Execute or inspect a restart-safe research experiment manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestration.durable_runner import DurableExperimentRunner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="YAML/JSON experiment manifest.")
    parser.add_argument("--state-dir", help="Persistent state directory; keep this on durable storage.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--job", action="append", dest="jobs", help="Run only this job (repeatable).")
    parser.add_argument("--stage", action="append", dest="stages", help="Run only this stage (repeatable).")
    parser.add_argument("--plan", action="store_true", help="Validate and print the DAG without executing it.")
    parser.add_argument("--plan-summary", action="store_true", help="Print compact stage/job counts without executing.")
    parser.add_argument(
        "--migrate-source",
        action="store_true",
        help="Audit and accept an operational source-only change before checkpoint resume.",
    )
    parser.add_argument("--migration-reason", help="Detailed reason for the source migration.")
    parser.add_argument(
        "--allow-source-path",
        action="append",
        default=[],
        help="Project-relative changed source path or glob allowed by the migration.",
    )
    parser.add_argument("--migration-operator", help="Person or automation approving the migration.")
    parser.add_argument(
        "--migrate-runtime",
        action="store_true",
        help="Audit and accept a safe host-runtime metadata change before checkpoint resume.",
    )
    parser.add_argument(
        "--allow-runtime-field",
        action="append",
        default=[],
        help="Changed top-level runtime field explicitly accepted by the migration.",
    )
    args = parser.parse_args()

    runner = DurableExperimentRunner(
        Path(args.manifest),
        Path(args.state_dir) if args.state_dir else None,
        Path(args.project_root),
    )
    selected_stages = set(args.stages) if args.stages else None
    if args.migrate_source and args.migrate_runtime:
        parser.error("Choose either --migrate-source or --migrate-runtime, not both.")
    if args.migrate_source:
        if args.plan or args.plan_summary or args.jobs or args.stages:
            parser.error("--migrate-source cannot be combined with plan, job, or stage execution.")
        if not args.migration_reason:
            parser.error("--migration-reason is required with --migrate-source.")
        result = runner.migrate_source_contract(
            reason=args.migration_reason,
            allowed_paths=args.allow_source_path,
            allowed_runtime_fields=args.allow_runtime_field,
            operator=args.migration_operator,
        )
    elif args.migrate_runtime:
        if args.plan or args.plan_summary or args.jobs or args.stages:
            parser.error("--migrate-runtime cannot be combined with plan, job, or stage execution.")
        if not args.migration_reason:
            parser.error("--migration-reason is required with --migrate-runtime.")
        result = runner.migrate_runtime_contract(
            reason=args.migration_reason,
            allowed_fields=args.allow_runtime_field,
            operator=args.migration_operator,
        )
    elif args.plan or args.plan_summary:
        plan = runner.plan(selected_stages)
        if args.plan_summary:
            result = {
                "run_id": runner.manifest.run_id,
                "total": len(plan),
                "completed": sum(item["status"] == "completed" for item in plan),
                "by_stage": {
                    stage: sum(item["stage"] == stage for item in plan)
                    for stage in sorted({item["stage"] for item in plan})
                },
                "declared_gpu_hour_upper_bound": sum(
                    item["estimated_hours"] for item in plan if item["uses_gpu"]
                ),
            }
        else:
            result = plan
    else:
        result = runner.run(set(args.jobs) if args.jobs else None, selected_stages)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
