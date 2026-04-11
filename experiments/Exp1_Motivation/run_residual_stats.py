from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from experiments.common import add_common_args, build_setup
from src.utils.runtime import write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect residual branch statistics.")
    add_common_args(parser)
    parser.add_argument("--output", default="results/motivation/residual_stats.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup = build_setup(args, compensator_name="identity")
    model = setup["model"]
    bundle = setup["bundle"]
    device = setup["device"]

    accumulators = defaultdict(lambda: {"ratio_sum": 0.0, "cos_sum": 0.0, "count": 0})
    model.eval()
    with torch.no_grad():
        for batch_index, (images, _) in enumerate(bundle.val_loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            images = images.to(device)
            output = model(images, mode="full", return_residual_stats=True)
            for block_name, stats in output["residual_stats"].items():
                plain = stats["plain"].flatten(1)
                identity = stats["identity"].flatten(1)
                ratio = plain.norm(dim=1) / (identity.norm(dim=1) + 1e-8)
                cosine = F.cosine_similarity(plain, identity, dim=1)
                accumulators[block_name]["ratio_sum"] += float(ratio.mean().item())
                accumulators[block_name]["cos_sum"] += float(cosine.mean().item())
                accumulators[block_name]["count"] += 1

    rows = []
    for block_name, stats in accumulators.items():
        count = max(stats["count"], 1)
        rows.append(
            {
                "dataset_source": bundle.source,
                "model": args.model,
                "block": block_name,
                "l2_ratio_mean": stats["ratio_sum"] / count,
                "cosine_mean": stats["cos_sum"] / count,
            }
        )
    output_path = write_csv(args.output, rows)
    print(f"Saved residual statistics to {output_path}")


if __name__ == "__main__":
    main()
