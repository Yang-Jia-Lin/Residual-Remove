"""Scripts/Exp4_Ablation/run_generalization.py"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.common import add_common_args, build_setup
from Src.Models_Training.calibrate import calibrate_compensators
from Src.Models_Training.trainer import evaluate_model
from Src.Utils.calibration import build_calibration_loader
from Src.Utils.runtime import write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generalization test across architectures.")
    add_common_args(parser)
    parser.add_argument("--models", default="resnet18,mobilenet_v2")
    parser.add_argument("--compensator", default="linear1x1")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--calib-size", type=int, default=None)
    parser.add_argument("--output", default="results/ablation/generalization.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_names = [item.strip() for item in args.models.split(",") if item.strip()]
    rows = []
    original_model_name = args.model

    for model_name in model_names:
        args.model = model_name
        setup = build_setup(args, compensator_name=args.compensator)
        configs = setup["configs"]
        model = setup["model"]
        bundle = setup["bundle"]
        device = setup["device"]
        blocks = model.get_block_names()
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
            removed_blocks=blocks,
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
            removed_blocks=blocks,
            max_batches=args.max_batches,
        )
        rows.append(
            {
                "dataset_source": bundle.source,
                "model": model_name,
                "compensator": args.compensator,
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "loss": metrics["loss"],
                "final_calibration_loss": history["epoch_loss"][-1],
            }
        )

    args.model = original_model_name
    output_path = write_csv(args.output, rows)
    print(f"Saved generalization results to {output_path}")


if __name__ == "__main__":
    main()
