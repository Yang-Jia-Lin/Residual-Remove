from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import add_common_args, build_setup
from src.training.calibrate import calibrate_compensators
from src.training.trainer import evaluate_model
from src.utils.calibration import build_calibration_loader
from src.utils.runtime import write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare full removal with local boundary removal.")
    add_common_args(parser)
    parser.add_argument("--compensator", default="linear1x1")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--calib-size", type=int, default=None)
    parser.add_argument("--boundary-count", type=int, default=2)
    parser.add_argument("--output", default="results/ablation/partial_removal.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup = build_setup(args, compensator_name=args.compensator)
    configs = setup["configs"]
    base_blocks = setup["model"].get_block_names()
    strategies = {
        "full": base_blocks,
        "front_boundary": base_blocks[: args.boundary_count],
        "back_boundary": base_blocks[-args.boundary_count :],
        "both_boundary": sorted(set(base_blocks[: args.boundary_count] + base_blocks[-args.boundary_count :])),
    }
    rows = []

    for name, removed_blocks in strategies.items():
        setup = build_setup(args, compensator_name=args.compensator)
        model = setup["model"]
        bundle = setup["bundle"]
        device = setup["device"]
        training_cfg = configs["compensator"]["training"]
        calibration_loader = build_calibration_loader(
            bundle.train_dataset,
            calib_size=args.calib_size or int(training_cfg["calib_size"]),
            batch_size=int(training_cfg["batch_size"]),
            seed=args.seed,
        )
        history = calibrate_compensators(
            model,
            calibration_loader=calibration_loader,
            device=device,
            removed_blocks=removed_blocks,
            epochs=args.epochs or int(training_cfg["epochs"]),
            lr=float(training_cfg["lr"]),
            weight_decay=float(training_cfg["weight_decay"]),
            feature_loss_weight=float(training_cfg["feature_loss_weight"]),
            logit_loss_weight=float(training_cfg["logit_loss_weight"]),
            grad_clip=float(training_cfg["grad_clip"]),
            max_batches=args.max_batches,
        )
        metrics = evaluate_model(
            model,
            bundle.val_loader,
            device=device,
            mode="compensated",
            removed_blocks=removed_blocks,
            max_batches=args.max_batches,
        )
        rows.append(
            {
                "dataset_source": bundle.source,
                "model": args.model,
                "strategy": name,
                "removed_count": len(removed_blocks),
                "removed_blocks": ",".join(removed_blocks),
                "chain_safe_blocks": len(removed_blocks),
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "loss": metrics["loss"],
                "final_calibration_loss": history["epoch_loss"][-1],
            }
        )

    output_path = write_csv(args.output, rows)
    print(f"Saved partial-removal ablation to {output_path}")


if __name__ == "__main__":
    main()
