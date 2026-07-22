"""Orchestrate Phase 1 experiments: baseline -> ablation -> analysis

This script runs the baseline training, then the detector-ablation training
with identical configs where possible, then runs the phase1 analysis and
generates filled markdown reports under reports/research_phase1/.
"""
from pathlib import Path
import subprocess
import sys
import json
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
BASE_CONFIG = REPO_ROOT / "configs" / "cycle_model.yaml"
ABLATION_CONFIG = REPO_ROOT / "configs" / "cycle_model_detector_ablation.yaml"
EXPERIMENT_REPORT_DIR = REPO_ROOT / "reports" / "experiments"
OUT_DIR = REPO_ROOT / "reports" / "research_phase1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _run_train(config_path: Path) -> None:
    cmd = [PY, str(REPO_ROOT / "src" / "trainers" / "train_cycle_model.py"), "--config-path", str(config_path)]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Training failed for config {config_path} (exit {res.returncode})")


def _latest_json_report() -> Path | None:
    files = sorted(EXPERIMENT_REPORT_DIR.glob("cycle_model_*.json"))
    return files[-1] if files else None


def _run_analysis():
    cmd = [PY, str(REPO_ROOT / "scripts" / "phase1_analysis.py")]
    print("Running analysis:", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError("phase1_analysis.py failed")


def _load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_md(path: Path, title: str, content_lines: list[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# " + title + "\n\n")
        f.write("\n".join(content_lines))


def main():
    # 1) baseline
    print("Starting baseline run with config:", BASE_CONFIG)
    _run_train(BASE_CONFIG)
    time.sleep(1)
    baseline_report = _latest_json_report()
    if baseline_report is None:
        raise RuntimeError("No baseline report found in reports/experiments after baseline run")
    print("Baseline report:", baseline_report)

    # 2) ablation
    print("Starting ablation run with config:", ABLATION_CONFIG)
    _run_train(ABLATION_CONFIG)
    time.sleep(1)
    ablation_report = _latest_json_report()
    if ablation_report is None or ablation_report == baseline_report:
        # try to find the second-latest file
        files = sorted(EXPERIMENT_REPORT_DIR.glob("cycle_model_*.json"))
        if len(files) >= 2:
            ablation_report = files[-1]
        else:
            raise RuntimeError("No ablation report found in reports/experiments after ablation run")
    print("Ablation report:", ablation_report)

    # 3) run analysis which writes reports/research_phase1/phase1_summary.json
    _run_analysis()
    analysis_json = OUT_DIR / "phase1_summary.json"
    if not analysis_json.exists():
        raise RuntimeError("phase1 analysis did not produce phase1_summary.json")

    # 4) Load reports and produce filled markdowns
    baseline_data = _load_json(baseline_report)
    ablation_data = _load_json(ablation_report)
    analysis_data = _load_json(analysis_json)

    # Fill experiment_1_detector_ablation.md
    exp1_lines = []
    exp1_lines.append(f"- Baseline report: {baseline_report.name}")
    exp1_lines.append(f"- Ablation report: {ablation_report.name}")
    exp1_lines.append("")
    exp1_lines.append("## Baseline Test Metrics (test split)")
    b_test = baseline_data.get("report", {}).get("test", {})
    for k in ["future_target_mae","future_target_rmse","future_target_pearson","future_target_r2","profitable_cycle_rate","catastrophic_cycle_rate"]:
        exp1_lines.append(f"- {k}: {b_test.get(k, 'NA')}")
    exp1_lines.append("")
    exp1_lines.append("## Ablation Test Metrics (test split)")
    a_test = ablation_data.get("report", {}).get("test", {})
    for k in ["future_target_mae","future_target_rmse","future_target_pearson","future_target_r2","profitable_cycle_rate","catastrophic_cycle_rate"]:
        exp1_lines.append(f"- {k}: {a_test.get(k, 'NA')}")
    _write_md(OUT_DIR / "experiment_1_detector_ablation.md", "Experiment 1: Detector Ablation", exp1_lines)

    # Fill experiment_2_latent_analysis.md
    exp2_lines = []
    exp2_lines.append(f"- Analysis file: {analysis_json.name}")
    exp2_lines.append("")
    neigh = analysis_data.get("neighbor", {})
    exp2_lines.append("## Neighbor similarity")
    for k, v in neigh.items():
        exp2_lines.append(f"- {k}: {v}")
    _write_md(OUT_DIR / "experiment_2_latent_analysis.md", "Experiment 2: Latent Analysis", exp2_lines)

    # Fill experiment_3_opportunity_ranking.md
    exp3_lines = []
    exp3_lines.append(f"- Analysis file: {analysis_json.name}")
    exp3_lines.append("")
    rank = analysis_data.get("ranking", {})
    exp3_lines.append("## Ranking results")
    for k, v in rank.items():
        exp3_lines.append(f"- {k}: {v}")
    _write_md(OUT_DIR / "experiment_3_opportunity_ranking.md", "Experiment 3: Opportunity Ranking", exp3_lines)

    # Fill summary.md with brief conclusions placeholders
    summary_lines = []
    summary_lines.append(f"Baseline report: {baseline_report.name}")
    summary_lines.append(f"Ablation report: {ablation_report.name}")
    summary_lines.append("")
    summary_lines.append("# Phase 1 Summary")
    summary_lines.append("")
    summary_lines.append("## 1) Is detector supervision helping?")
    summary_lines.append("- Evidence: see experiment_1_detector_ablation.md")
    summary_lines.append("")
    summary_lines.append("## 2) Does latent space contain future information?")
    summary_lines.append("- Evidence: see experiment_2_latent_analysis.md")
    summary_lines.append("")
    summary_lines.append("## 3) Can the model rank future leaders?")
    summary_lines.append("- Evidence: see experiment_3_opportunity_ranking.md")
    summary_lines.append("")
    summary_lines.append("## Raw analysis summary")
    summary_lines.append(json.dumps(analysis_data, indent=2))
    _write_md(OUT_DIR / "summary.md", "Phase 1 Summary", summary_lines)

    print("Orchestration complete. Reports written to:", OUT_DIR)


if __name__ == "__main__":
    main()

