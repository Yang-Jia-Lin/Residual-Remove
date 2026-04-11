from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import add_common_args, build_setup
from src.training.trainer import evaluate_model
from src.utils.runtime import write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure accuracy drop under progressive residual removal.")
    add_common_args(parser)
    parser.add_argument("--output", default="results/motivation/acc_drop.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup = build_setup(args, compensator_name="identity")
    model = setup["model"]
    bundle = setup["bundle"]
    device = setup["device"]
    blocks = model.get_block_names()

    rows: list[dict[str, float | int | str]] = []
    full_metrics = evaluate_model(model, bundle.val_loader, device=device, mode="full", max_batches=args.max_batches)
    rows.append(
        {
            "dataset_source": bundle.source,
            "model": args.model,
            "mode": "full",
            "removed_count": 0,
            "removed_blocks": "",
            "top1": full_metrics["top1"],
            "top5": full_metrics["top5"],
            "loss": full_metrics["loss"],
            "top1_drop": 0.0,
        }
    )

    for remove_count in range(1, len(blocks) + 1):
        removed = blocks[-remove_count:]
        metrics = evaluate_model(
            model,
            bundle.val_loader,
            device=device,
            mode="plain",
            removed_blocks=removed,
            max_batches=args.max_batches,
        )
        rows.append(
            {
                "dataset_source": bundle.source,
                "model": args.model,
                "mode": "plain",
                "removed_count": remove_count,
                "removed_blocks": ",".join(removed),
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "loss": metrics["loss"],
                "top1_drop": full_metrics["top1"] - metrics["top1"],
            }
        )

    output_path = write_csv(args.output, rows)
    print(f"Saved accuracy-drop results to {output_path}")


if __name__ == "__main__":
    main()
