import json
from pathlib import Path

import yaml

import scripts.run_temporal_transport_study as study_module


def test_encoder_config_freezes_runtime_memory_and_enables_safe_amp(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(study_module, "PROJECT_ROOT", tmp_path)
    base = {
        "data": {"precomputed_dir": "unused"},
        "model": {},
        "training": {},
        "split": {},
        "features": {"sequence": [], "future_targets": []},
        "latent_export": {},
        "evaluation": {},
    }
    study = {
        "data": {
            "root": "data/international",
            "markets": ["US", "India"],
            "market_balanced_sampling": True,
            "relative_outcomes": {"enabled": True},
            "cross_sectional_relative_features": {
                "tech_return_1": "relative_return_1"
            },
        },
        "training": {
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "validation_selection_metric": "future_target_mae",
            "validation_selection_mode": "min",
            "use_amp": True,
            "amp_dtype": "auto",
            "amp_forward_fallback_to_fp32": True,
            "checkpoint_interval_steps": 10,
        },
        "periods": {
            "train_years": 2,
            "validation_years": 1,
            "test_years": 1,
            "selected_fold": 0,
            "observed": {"start": "2025-01-01", "end": "2025-03-31"},
        },
        "variants": {"candidate": {"loss": {"lambda_reg": 1.0}}},
        "hardware": {"vram_fraction": 0.9},
    }

    config_path, _, _ = study_module._encoder_config(
        base, study, "candidate", 7, tmp_path / "run"
    )
    generated = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert generated["model"]["memory_mode"] == "static_parameter"
    assert generated["model"]["memory_update_rate"] == 0.0
    assert generated["training"]["amp_dtype"] == "auto"
    assert generated["training"]["amp_forward_fallback_to_fp32"] is True


def test_compare_gates_only_on_development_test(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(study_module, "PROJECT_ROOT", tmp_path)
    config = {
        "study": {"output_root": "reports/transport"},
        "training": {"seeds": [7, 17, 37]},
        "variants": {"baseline": {}, "candidate": {}},
        "comparison": {
            "baseline": "baseline",
            "candidate": "candidate",
            "minimum_seed_wins": 2,
            "minimum_mean_sharpe_delta": 0.0,
            "minimum_worst_seed_sharpe_delta": 0.0,
            "maximum_candidate_drawdown": 0.25,
        },
    }
    config_path = tmp_path / "study.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    for period in ("test", "observed"):
        for variant in config["variants"]:
            for seed in config["training"]["seeds"]:
                path = (
                    tmp_path
                    / "reports/transport/run"
                    / "memory"
                    / f"{variant}_seed_{seed}"
                    / period
                    / "eval/metrics.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                baseline_sharpe = 0.1
                candidate_sharpe = 0.2 if period == "test" else -0.5
                metrics = {
                    "total_return": candidate_sharpe if variant == "candidate" else 0.1,
                    "annualized_return": 0.1,
                    "sharpe": candidate_sharpe if variant == "candidate" else baseline_sharpe,
                    "max_drawdown": -0.1,
                    "trade_count": 10,
                    "score_realized_return_spearman": 0.1,
                }
                path.write_text(json.dumps(metrics), encoding="utf-8")

    verdict = study_module.compare(config_path, "run")

    assert verdict["candidate_passed_development_gate"] is True
    assert verdict["development_seed_wins"] == 3
    assert verdict["observed_seed_wins"] == 0
    assert verdict["promotion_allowed"] is False
