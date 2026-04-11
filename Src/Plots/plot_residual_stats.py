from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot residual statistics.")
    parser.add_argument("--input", default="results/motivation/residual_stats.csv")
    parser.add_argument("--output", default="results/motivation/residual_stats.png")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    plt.figure(figsize=(10, 5))
    plt.scatter(frame["l2_ratio_mean"], frame["cosine_mean"])
    for _, row in frame.iterrows():
        plt.annotate(row["block"], (row["l2_ratio_mean"], row["cosine_mean"]), fontsize=7)
    plt.xlabel("Mean ||F(x)|| / ||x||")
    plt.ylabel("Mean Cosine Similarity")
    plt.title("Residual Signal Statistics")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
