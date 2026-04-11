from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import add_common_args, build_setup, get_probe_batch
from src.system.bandwidth_sim import estimate_transfer_time_ms
from src.utils.runtime import tensor_bytes, write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate DAG vs chain transfer cost around residual blocks.")
    add_common_args(parser)
    parser.add_argument("--output", default="results/motivation/system_cost.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup = build_setup(args, compensator_name="identity")
    model = setup["model"]
    configs = setup["configs"]
    system_cfg = configs["system"]
    device = setup["device"]
    bundle = setup["bundle"]

    images, _ = get_probe_batch(bundle, device)
    output = model(images, mode="full", return_residual_stats=True)

    rows = []
    for block_name, stats in output["residual_stats"].items():
        dag_bytes = tensor_bytes(stats["plain"]) + tensor_bytes(stats["identity"])
        chain_bytes = tensor_bytes(stats["plain"])
        for bandwidth in system_cfg["bandwidth_mbps"]:
            dag_ms = estimate_transfer_time_ms(
                dag_bytes,
                bandwidth_mbps=float(bandwidth),
                protocol_overhead_ms=float(system_cfg["protocol_overhead_ms"]),
            )
            chain_ms = estimate_transfer_time_ms(
                chain_bytes,
                bandwidth_mbps=float(bandwidth),
                protocol_overhead_ms=float(system_cfg["protocol_overhead_ms"]),
            )
            rows.append(
                {
                    "dataset_source": bundle.source,
                    "model": args.model,
                    "block": block_name,
                    "bandwidth_mbps": float(bandwidth),
                    "dag_bytes": dag_bytes,
                    "chain_bytes": chain_bytes,
                    "saved_bytes": dag_bytes - chain_bytes,
                    "dag_transfer_ms": dag_ms,
                    "chain_transfer_ms": chain_ms,
                    "saved_ms": dag_ms - chain_ms,
                }
            )

    output_path = write_csv(args.output, rows)
    print(f"Saved system-cost results to {output_path}")


if __name__ == "__main__":
    main()
