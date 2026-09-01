"""CLI tool to extract, seal, and export a complete Research Defense Evidence Package.

Gathers all provenance, causality proofs, target lineage, universe accounting,
primary system comparisons (P0–P6), bootstrap significance tests with FDR/Holm,
CSCV PBO, trade ledgers, publication LaTeX tables, and cryptographic SHA-256 seals.

Usage:
  python scripts/extract_research_defense_package.py
  python scripts/extract_research_defense_package.py --run-root reports/reconstruction_v1 --format all
  python scripts/extract_research_defense_package.py --output exports/manuscript_defense_v1.tar.gz --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.research_defense_extractor import DefensePackageBuilder, export_research_defense_bundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_defense_package")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and package full research defense and manuscript evidence bundle."
    )
    parser.add_argument(
        "--run-root",
        action="append",
        default=None,
        help="One or more experiment run root directories to scan (e.g. reports/reconstruction_v1).",
    )
    parser.add_argument(
        "--output",
        default="reports/reconstruction_v1/research_defense_bundle.tar.gz",
        help="Output archive or directory path.",
    )
    parser.add_argument(
        "--format",
        choices=["tar.gz", "zip", "dir", "all"],
        default="all",
        help="Export format: 'tar.gz', 'zip', 'dir', or 'all' (default: all).",
    )
    parser.add_argument(
        "--profile",
        choices=["reviewer", "full"],
        default="reviewer",
        help="Artifact collection profile (default: reviewer).",
    )
    parser.add_argument(
        "--title",
        default="Causal Market Memory Reconstruction",
        help="Document and report title.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Automatically verify cryptographic integrity of the generated bundle.",
    )
    args = parser.parse_args()

    run_roots = args.run_root or ["reports/reconstruction_v1"]
    resolved_roots = []
    for r in run_roots:
        p = Path(r)
        resolved = (PROJECT_ROOT / p).resolve() if not p.is_absolute() else p.resolve()
        resolved_roots.append(resolved)

    logger.info("Starting Research Defense Package export...")
    logger.info("Run roots: %s", [str(r) for r in resolved_roots])
    logger.info("Output destination: %s", args.output)
    logger.info("Export format: %s | Profile: %s", args.format, args.profile)

    builder = DefensePackageBuilder(
        run_roots=resolved_roots,
        output_path=args.output,
        profile=args.profile,
        bundle_title=args.title,
        project_root=PROJECT_ROOT,
    )

    result = builder.export_archive(export_format=args.format)

    print("\n" + "=" * 70)
    print(" RESEARCH DEFENSE EVIDENCE PACKAGE EXPORT COMPLETED")
    print("=" * 70)
    print(f"Status:        {result.get('status')}")
    print(f"Primary Output: {result.get('output')}")
    print(f"Format:        {result.get('format')}")
    print(f"Files Sealed:  {result.get('file_count')}")
    print(f"Payload Size:  {result.get('payload_bytes', 0):,} bytes")
    if result.get("sha256"):
        print(f"Archive SHA256: {result.get('sha256')}")
    print("=" * 70 + "\n")

    if args.verify:
        logger.info("Running automatic cryptographic verification...")
        # If directory was exported or archive extracted, run verify_bundle.py
        target_dir = Path(result["output"]).parent / Path(result["output"]).stem.split(".")[0]
        verify_script = target_dir / "verify_bundle.py"
        if verify_script.exists():
            import subprocess
            res = subprocess.run([sys.executable, str(verify_script)], capture_output=True, text=True)
            print(res.stdout)
            if res.returncode != 0:
                logger.error("Verification failed: %s", res.stderr)
                sys.exit(1)
            logger.info("Cryptographic verification PASSED successfully.")

    # Save export receipt
    receipt_file = Path(args.output).parent / "export_receipt.json"
    receipt_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Saved export receipt to %s", receipt_file)


if __name__ == "__main__":
    main()
