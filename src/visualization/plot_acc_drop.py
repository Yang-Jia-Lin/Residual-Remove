from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot accuracy drop curve.")
    parser.add_argument("--input", default="results/motivation/acc_drop.csv")
    parser.add_argument("--output", default="results/motivation/acc_drop.png")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    frame = frame.sort_values("removed_count")
    plt.figure(figsize=(8, 5))
    plt.plot(frame["removed_count"], frame["top1"], marker="o")
    plt.xlabel("Removed Residual Blocks")
    plt.ylabel("Top-1 Accuracy")
    plt.title("Accuracy Drop Under Progressive Residual Removal")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
