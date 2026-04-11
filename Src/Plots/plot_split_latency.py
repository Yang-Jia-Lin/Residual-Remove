from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot stacked split-inference latency.")
    parser.add_argument("--input", default="results/system/split_inference.csv")
    parser.add_argument("--output", default="results/system/split_latency.png")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    frame["label"] = frame["topology"] + "@" + frame["bandwidth_mbps"].astype(str) + "Mbps"
    plt.figure(figsize=(11, 5))
    plt.bar(frame["label"], frame["edge_ms"], label="Edge")
    plt.bar(frame["label"], frame["transfer_ms"], bottom=frame["edge_ms"], label="Transfer")
    plt.bar(
        frame["label"],
        frame["cloud_ms"],
        bottom=frame["edge_ms"] + frame["transfer_ms"],
        label="Cloud",
    )
    plt.ylabel("Latency (ms)")
    plt.title("Split Inference Latency Breakdown")
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
