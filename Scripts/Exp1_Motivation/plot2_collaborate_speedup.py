"""Scripts/Exp1_Motivation/plot_collaborate_latency.py"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from Configs.paras import COLORS
from Src.Utils.plot_utils import save_fig_for_ieee, set_ieee_style

_BW_COLORS = [COLORS["blue"], COLORS["red"], COLORS["green"], COLORS["purple"]]
_MARKER_DAG = "o"
_MARKER_CHAIN = "s"


def plot_collab_cost(rows: list[dict], save_path: Path) -> None:
    # 数据
    bandwidths = sorted(set(float(r["bandwidth_mbps"]) for r in rows))
    seen, block_names = set(), []
    for r in rows:
        if r["block"] not in seen:
            seen.add(r["block"])
            block_names.append(r["block"])
    block_to_idx = {name: i for i, name in enumerate(block_names)}

    # 画布
    set_ieee_style(mode="single")
    fig, ax = plt.subplots()

    # ── 每种带宽画两条线（DAG 实线 / Chain 虚线） ─────────────────────────────
    for i, bw in enumerate(bandwidths):
        bw_rows = sorted(
            [r for r in rows if float(r["bandwidth_mbps"]) == bw],
            key=lambda r: block_to_idx[r["block"]],
        )
        xi = np.array([block_to_idx[r["block"]] + 1 for r in bw_rows])
        y_dag = np.array([float(r["dag_total_ms"]) for r in bw_rows])
        y_chain = np.array([float(r["chain_total_ms"]) for r in bw_rows])

        color = _BW_COLORS[i % len(_BW_COLORS)]
        label = f"{int(bw)} Mbps" if bw == int(bw) else f"{bw} Mbps"

        ax.plot(
            xi,
            y_dag,
            color=color,
            marker=_MARKER_DAG,
            linestyle="-",
            linewidth=1.5,
            label=f"{label} – DAG",
            zorder=3,
        )
        ax.plot(
            xi,
            y_chain,
            color=color,
            marker=_MARKER_CHAIN,
            linestyle="--",
            linewidth=1.5,
            label=f"{label} – Chain",
            zorder=3,
            alpha=0.6,
        )

    # ── 坐标轴 ────────────────────────────────────────────────────────────────
    ax.set_yscale("log")  # <--- 新增：将 Y 轴设置为对数刻度 (10^0, 10^1, 10^2...)

    ax.set_xlabel("Split Point Position (Block Index)")
    ax.set_ylabel("Transmission Latency (ms) [Log]")
    ax.set_xlim(left=0, right=len(block_names) + 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 图例
    ax.legend(
        loc="lower center",  # 以图例的“中下部”为锚点
        bbox_to_anchor=(
            0.5,
            1.02,
        ),  # 锚点放在坐标轴的 (x=50%, y=102%) 也就是正上方偏外一点
        ncol=3,
        fontsize=8,  # 字体可以稍微缩小一点点
        frameon=False,  # 放在外面的图例通常建议去掉边框，看起来更干净融合
        handlelength=1.5,
        handletextpad=0.3,
        columnspacing=0.8,
        labelspacing=0.3,
    )

    # ── 标题 ──────────────────────────────────────────────────────────────────
    # ax.set_title("Collaborative Inference Transmission Cost")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    save_fig_for_ieee(save_path, fig)
    plt.close(fig)


def plot_from_csv(csv_path: Path) -> None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    save_path = csv_path.with_name(csv_path.stem + "_plot")
    plot_collab_cost(rows, save_path)


if __name__ == "__main__":
    example_csv = Path(
        "/root/autodl-tmp/ResidualRemove/Results/Exp1_Motivation/Motivation2_Collaborate_cost/20260428_1105_collaborate.csv"
    )
    plot_from_csv(example_csv)
