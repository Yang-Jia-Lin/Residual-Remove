"""Scripts/Exp3_System/run_concurrent.py"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.common import add_common_args, build_setup
from Src.Models_Evaluation.latency import measure_latency
from Src.Models_Evaluation.memory import measure_peak_memory
from Src.Models_Training.calibrate import calibrate_compensators
from Src.Utils.calibration import build_calibration_loader
from Src.Utils.runtime import write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate multi-tenant throughput under a memory budget.")
    add_common_args(parser)
    parser.add_argument("--compensator", default="affine")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--calib-size", type=int, default=None)
    parser.add_argument("--output", default="results/system/concurrent.csv")
    return parser


def _profile_variant(args, mode_name: str, compensator_name: str) -> dict[str, float | str]:
    setup = build_setup(args, compensator_name=compensator_name)
    configs = setup["configs"]
    model = setup["model"]
    bundle = setup["bundle"]
    device = setup["device"]
    blocks = model.get_block_names()
    training_cfg = configs["compensator"]["training"]
    system_cfg = configs["system"]

    if mode_name == "compensated":
        calibration_loader = build_calibration_loader(
            bundle.train_dataset,
            calib_size=args.calib_size or int(training_cfg["calib_size"]),
            batch_size=int(training_cfg["batch_size"]),
            seed=args.seed,
        )
        calibrate_compensators(
            model,
            calibration_loader=calibration_loader,
            device=device,
            removed_blocks=blocks,
            epochs=args.epochs or int(training_cfg["epochs"]),
            lr=float(training_cfg["lr"]),
            weight_decay=float(training_cfg["weight_decay"]),
            feature_loss_weight=float(training_cfg["feature_loss_weight"]),
            logit_loss_weight=float(training_cfg["logit_loss_weight"]),
            grad_clip=float(training_cfg["grad_clip"]),
            max_batches=args.max_batches,
        )

    images, _ = next(iter(bundle.val_loader))
    images = images.to(device)
    forward_kwargs = {"mode": mode_name} if mode_name == "full" else {"mode": mode_name, "removed_blocks": blocks}
    latency_ms = measure_latency(model, images, **forward_kwargs)
    memory = measure_peak_memory(model, images, **forward_kwargs)
    budget_bytes = float(system_cfg["edge_memory_budget_mb"]) * 1024 * 1024
    peak_bytes = max(memory["peak_bytes"], 1.0)
    max_concurrency = max(1, int(budget_bytes // peak_bytes))
    qps = max_concurrency * 1000.0 / max(latency_ms, 1e-6)
    return {
        "dataset_source": bundle.source,
        "model": args.model,
        "mode": mode_name,
        "latency_ms": latency_ms,
        "peak_memory_mb": memory["peak_mb"],
        "memory_budget_mb": float(system_cfg["edge_memory_budget_mb"]),
        "max_concurrency": max_concurrency,
        "estimated_qps": qps,
    }


def main() -> None:
    args = build_parser().parse_args()
    rows = [
        _profile_variant(args, mode_name="full", compensator_name="identity"),
        _profile_variant(args, mode_name="plain", compensator_name="identity"),
        _profile_variant(args, mode_name="compensated", compensator_name=args.compensator),
    ]
    output_path = write_csv(args.output, rows)
    print(f"Saved concurrency simulation to {output_path}")


if __name__ == "__main__":
    main()
