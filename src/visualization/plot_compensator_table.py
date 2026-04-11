from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot compensator benchmark bars.")
    parser.add_argument("--input", default="results/compensator/benchmark.csv")
    parser.add_argument("--output", default="results/compensator/benchmark_top1.png")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    plt.figure(figsize=(9, 5))
    plt.bar(frame["compensator"], frame["top1"])
    plt.ylabel("Top-1 Accuracy")
    plt.title("Compensator Benchmark")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
