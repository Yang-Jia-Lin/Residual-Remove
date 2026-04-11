from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import add_common_args, build_setup
from src.system.split_runner import run_split_inference
from src.training.calibrate import calibrate_compensators
from src.utils.calibration import build_calibration_loader
from src.utils.runtime import write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run split-inference simulation.")
    add_common_args(parser)
    parser.add_argument("--compensator", default="affine")
    parser.add_argument("--chain-mode", choices=["plain", "compensated"], default="plain")
    parser.add_argument("--split-point", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--calib-size", type=int, default=None)
    parser.add_argument("--output", default="results/system/split_inference.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup = build_setup(args, compensator_name=args.compensator if args.chain_mode == "compensated" else "identity")
    configs = setup["configs"]
    model = setup["model"]
    bundle = setup["bundle"]
    device = setup["device"]
    system_cfg = configs["system"]
    training_cfg = configs["compensator"]["training"]
    blocks = model.get_block_names()

    if args.chain_mode == "compensated":
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
    split_point = args.split_point or blocks[len(blocks) // 2]
    rows = []

    for bandwidth in system_cfg["bandwidth_mbps"]:
        full_report = run_split_inference(
            model,
            images,
            split_point=split_point,
            bandwidth_mbps=float(bandwidth),
            protocol_overhead_ms=float(system_cfg["protocol_overhead_ms"]),
            compress=bool(system_cfg["compression"]["enabled"]),
            compression_method=str(system_cfg["compression"]["method"]),
            mode="full",
        )
        rows.append(
            {
                "dataset_source": bundle.source,
                "model": args.model,
                "topology": "full_residual",
                "split_point": split_point,
                "bandwidth_mbps": float(bandwidth),
                **full_report,
            }
        )

        chain_report = run_split_inference(
            model,
            images,
            split_point=split_point,
            bandwidth_mbps=float(bandwidth),
            protocol_overhead_ms=float(system_cfg["protocol_overhead_ms"]),
            compress=bool(system_cfg["compression"]["enabled"]),
            compression_method=str(system_cfg["compression"]["method"]),
            mode=args.chain_mode,
            removed_blocks=blocks,
        )
        rows.append(
            {
                "dataset_source": bundle.source,
                "model": args.model,
                "topology": f"chain_{args.chain_mode}",
                "split_point": split_point,
                "bandwidth_mbps": float(bandwidth),
                **chain_report,
            }
        )

    output_path = write_csv(args.output, rows)
    print(f"Saved split-inference simulation to {output_path}")


if __name__ == "__main__":
    main()
