"""Run the one permitted, gated Phase 5C transformer alignment pass."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.adapter import DecisionAdapter  # noqa: E402
from src.decision.alignment import (  # noqa: E402
    AlignedDecisionModel,
    AlignmentConfig,
    alignment_objectives,
    configure_alignment_parameters,
    pcgrad_backward,
    save_aligned_checkpoint,
    target_indices,
)
from src.decision.losses import DecisionLossConfig  # noqa: E402
from src.trainers.train_cycle_model import (  # noqa: E402
    load_checkpoint,
    load_precomputed_frame,
    make_datasets,
    make_loader,
)


def _evaluate(model, loader, device, indices, loss_cfg):
    model.eval()
    future_errors, decision_losses = [], []
    with torch.no_grad():
        for batch in loader:
            targets = batch["future_target"].to(device)
            outputs = model(batch["sequence"].to(device))
            outputs["quantiles"] = model.adapter.config.quantiles
            future, decision, _ = alignment_objectives(
                outputs, targets, list(batch["timestamp"]), list(batch["ticker"]), indices, loss_cfg
            )
            future_errors.append(float(future)); decision_losses.append(float(decision))
    return {"future_mae": float(np.mean(future_errors)), "decision_loss": float(np.mean(decision_losses))}


def run(base_checkpoint: str, base_config_path: str, adapter_checkpoint: str, output: str, config_path: str) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    align_cfg = AlignmentConfig(**config.get("alignment", {}))
    loss_cfg = DecisionLossConfig(**config["loss"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, payload = load_checkpoint(base_checkpoint, device)
    if not hasattr(encoder, "memory_mode"):
        raise TypeError("Phase 5C alignment requires the patch transformer.")
    encoder.memory_mode = "static_parameter"
    encoder.enable_memory_update = False
    encoder.initialise_static_memory_from_runtime()
    adapter_payload = torch.load(adapter_checkpoint, map_location=device)
    adapter = DecisionAdapter.from_checkpoint(adapter_payload)
    model = AlignedDecisionModel(encoder, adapter).to(device)
    encoder_params, adapter_params = configure_alignment_parameters(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": align_cfg.encoder_learning_rate},
            {"params": adapter_params, "lr": align_cfg.adapter_learning_rate},
        ],
        weight_decay=align_cfg.weight_decay,
    )
    base_config = yaml.safe_load(Path(base_config_path).read_text(encoding="utf-8"))
    frame = load_precomputed_frame(base_config)
    _, datasets, _, split_meta = make_datasets(frame, base_config)
    batch_size = int(base_config["training"]["batch_size"])
    train_loader = make_loader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = make_loader(datasets["val"], batch_size=batch_size, shuffle=False)
    indices = target_indices(list(datasets["train"].future_target_cols))
    baseline = _evaluate(model, val_loader, device, indices, loss_cfg)
    history = []
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for epoch in range(align_cfg.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            targets = batch["future_target"].to(device)
            if not torch.isfinite(targets).all():
                continue
            outputs = model(batch["sequence"].to(device))
            outputs["quantiles"] = model.adapter.config.quantiles
            future, decision, _ = alignment_objectives(
                outputs, targets, list(batch["timestamp"]), list(batch["ticker"]), indices, loss_cfg
            )
            optimizer.zero_grad(set_to_none=True)
            pcgrad_backward((future, decision), parameters)
            torch.nn.utils.clip_grad_norm_(parameters, align_cfg.gradient_clip)
            optimizer.step()
            losses.append(float(future + decision))
        validation = _evaluate(model, val_loader, device, indices, loss_cfg)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation": validation})
    final = history[-1]["validation"]
    future_preserved = final["future_mae"] <= baseline["future_mae"] * (1.0 + align_cfg.maximum_future_mae_regression)
    metadata = {
        "alignment_config": asdict(align_cfg),
        "baseline_validation": baseline,
        "final_validation": final,
        "future_preserved": future_preserved,
        "history": history,
        "split_meta": split_meta,
        "promotion_status": "eligible_for_backtest" if future_preserved else "rejected",
    }
    save_aligned_checkpoint(output, model, metadata)
    Path(output).with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/phase5_decision_alignment.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.base_checkpoint, args.base_config, args.adapter_checkpoint, args.output, args.config), indent=2, default=str))


if __name__ == "__main__":
    main()
