from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "ablation_presets.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preset ablation experiments.")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output-root", default=str(ROOT / "results"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for model_name in config["models"]:
        for ablation in config["ablations"]:
            exp_name = f"{model_name}_{ablation['name']}"
            output_dir = output_root / exp_name
            cmd = [
                args.python,
                str(ROOT / "scripts" / "train_eval.py"),
                "--model",
                model_name,
                "--dataset",
                args.dataset,
                "--data-root",
                args.data_root,
                "--output-dir",
                str(output_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--seed",
                str(args.seed),
                "--ablation-mode",
                ablation["mode"],
                "--ablation-value",
                str(ablation["value"]),
            ]
            print(f"Running {exp_name}")
            subprocess.run(cmd, check=True)

            metrics_path = output_dir / "metrics.json"
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            summary_rows.append(
                {
                    "experiment": exp_name,
                    "model": model_name,
                    "ablation_mode": ablation["mode"],
                    "ablation_value": ablation["value"],
                    "removed_ratio": metrics["block_summary"]["removed_ratio"],
                    "val_top1": metrics["final_train_eval"].get("val_top1", ""),
                    "peak_memory_mb": metrics["peak_memory"].get("peak_memory_mb", ""),
                    "latency_ms": metrics["latency"].get("latency_ms", ""),
                    "throughput_samples_per_sec": metrics["latency"].get(
                        "throughput_samples_per_sec", ""
                    ),
                    "lifetime_proxy_elements_x_ops": metrics["activation_lifetime_proxy"].get(
                        "lifetime_proxy_elements_x_ops", ""
                    ),
                }
            )

    summary_path = output_root / "summary.csv"
    fieldnames = [
        "experiment",
        "model",
        "ablation_mode",
        "ablation_value",
        "removed_ratio",
        "val_top1",
        "peak_memory_mb",
        "latency_ms",
        "throughput_samples_per_sec",
        "lifetime_proxy_elements_x_ops",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
