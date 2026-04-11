from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import add_common_args, build_setup, resolve_removed_blocks
from src.evaluation.energy import estimate_energy_joules
from src.evaluation.flops import count_parameters, estimate_macs
from src.evaluation.latency import measure_latency
from src.evaluation.memory import measure_peak_memory
from src.training.calibrate import calibrate_compensators
from src.training.trainer import evaluate_model
from src.utils.calibration import build_calibration_loader
from src.utils.runtime import write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark compensators under the same calibration setting.")
    add_common_args(parser)
    parser.add_argument("--compensators", default="plain,scalar,affine,linear1x1,low_rank,adapter")
    parser.add_argument("--output", default="results/compensator/benchmark.csv")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--calib-size", type=int, default=None)
    parser.add_argument("--removed-blocks", default="all")
    return parser


def _evaluate_variant(args, name: str) -> dict[str, float | str]:
    compensator_name = "identity" if name == "plain" else name
    setup = build_setup(args, compensator_name=compensator_name)
    configs = setup["configs"]
    model = setup["model"]
    bundle = setup["bundle"]
    device = setup["device"]
    blocks = resolve_removed_blocks(args.removed_blocks, model.get_block_names())

    training_cfg = configs["compensator"]["training"]
    system_cfg = configs["system"]
    calib_size = args.calib_size or int(training_cfg["calib_size"])
    epochs = args.epochs or int(training_cfg["epochs"])
    calibration_loader = build_calibration_loader(
        bundle.train_dataset,
        calib_size=calib_size,
        batch_size=int(training_cfg["batch_size"]),
        seed=args.seed,
    )

    eval_mode = "plain" if name == "plain" else "compensated"
    calibration_history = {"epoch_loss": [0.0]}
    if name != "plain":
        calibration_history = calibrate_compensators(
            model,
            calibration_loader=calibration_loader,
            device=device,
            removed_blocks=blocks,
            epochs=epochs,
            lr=float(training_cfg["lr"]),
            weight_decay=float(training_cfg["weight_decay"]),
            feature_loss_weight=float(training_cfg["feature_loss_weight"]),
            logit_loss_weight=float(training_cfg["logit_loss_weight"]),
            grad_clip=float(training_cfg["grad_clip"]),
            max_batches=args.max_batches,
        )

    sample, _ = next(iter(bundle.val_loader))
    sample = sample.to(device)
    metrics = evaluate_model(
        model,
        bundle.val_loader,
        device=device,
        mode=eval_mode,
        removed_blocks=blocks,
        max_batches=args.max_batches,
    )
    latency_ms = measure_latency(model, sample, mode=eval_mode, removed_blocks=blocks)
    memory = measure_peak_memory(model, sample, mode=eval_mode, removed_blocks=blocks)
    macs = estimate_macs(model, sample, mode=eval_mode, removed_blocks=blocks)
    params = count_parameters(model)
    energy = estimate_energy_joules(
        macs=macs,
        dram_bytes=int(memory["peak_bytes"]),
        dram_pj_per_byte=float(system_cfg["dram_pj_per_byte"]),
        mac_pj=float(system_cfg["mac_pj"]),
    )
    return {
        "dataset_source": bundle.source,
        "model": args.model,
        "compensator": name,
        "top1": metrics["top1"],
        "top5": metrics["top5"],
        "loss": metrics["loss"],
        "latency_ms": latency_ms,
        "peak_memory_mb": memory["peak_mb"],
        "macs": macs,
        "params": params,
        "energy_total_j": energy["total_j"],
        "final_calibration_loss": calibration_history["epoch_loss"][-1],
    }


def main() -> None:
    args = build_parser().parse_args()
    rows: list[dict[str, float | str]] = []

    full_setup = build_setup(args, compensator_name="identity")
    full_model = full_setup["model"]
    full_bundle = full_setup["bundle"]
    full_device = full_setup["device"]
    full_blocks = full_model.get_block_names()
    full_sample, _ = next(iter(full_bundle.val_loader))
    full_sample = full_sample.to(full_device)
    full_metrics = evaluate_model(
        full_model,
        full_bundle.val_loader,
        device=full_device,
        mode="full",
        max_batches=args.max_batches,
    )
    rows.append(
        {
            "dataset_source": full_bundle.source,
            "model": args.model,
            "compensator": "full_residual",
            "top1": full_metrics["top1"],
            "top5": full_metrics["top5"],
            "loss": full_metrics["loss"],
            "latency_ms": measure_latency(full_model, full_sample, mode="full"),
            "peak_memory_mb": measure_peak_memory(full_model, full_sample, mode="full")["peak_mb"],
            "macs": estimate_macs(full_model, full_sample, mode="full"),
            "params": count_parameters(full_model),
            "energy_total_j": 0.0,
            "final_calibration_loss": 0.0,
        }
    )

    for name in [item.strip() for item in args.compensators.split(",") if item.strip()]:
        rows.append(_evaluate_variant(args, name))

    output_path = write_csv(args.output, rows)
    print(f"Saved compensator benchmark to {output_path}")


if __name__ == "__main__":
    main()
