"""Src/Plots/plot_inference_tradeoff.py"""

import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from Configs.paras import COLORS
from Src.Utils.plot_utils import save_fig_for_ieee, set_ieee_style

# ── 调色 ──────────────────────────────────────────────────────────────────────
_COLOR_LAT = COLORS["blue"]  # 蓝：延迟
_COLOR_ACC = COLORS["red"]  # 红：精度
_MARKER_LAT = "o"
_MARKER_ACC = "s"


def plot_inference_tradeoff(rows: Sequence[dict], save_path: Path):
    # 提取数据
    x = [r["removed_count"] for r in rows]
    latency_ms = [r["latency_ms"] for r in rows]
    acc_top1 = [r["acc_top1"] for r in rows]

    # 画布
    set_ieee_style(mode="single")
    fig, ax_lat = plt.subplots()

    # 左轴：延迟
    ln_lat = ax_lat.plot(
        x,
        latency_ms,
        color=_COLOR_LAT,
        marker=_MARKER_LAT,
        label="Latency (ms)",
        zorder=3,
    )
    ax_lat.set_xlabel("Number of Removed Residual Blocks")
    ax_lat.set_ylabel("Inference Latency (ms)", color=_COLOR_LAT)
    ax_lat.tick_params(axis="y", labelcolor=_COLOR_LAT)
    ax_lat.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ── 右轴：精度 ────────────────────────────────────────────────────────────
    ax_acc = ax_lat.twinx()
    ln_acc = ax_acc.plot(
        x,
        acc_top1,
        color=_COLOR_ACC,
        marker=_MARKER_ACC,
        linestyle="--",
        label="Top-1 Acc (%)",
        zorder=3,
    )
    ax_acc.set_ylabel("Top-1 Accuracy (%)", color=_COLOR_ACC)
    ax_acc.tick_params(axis="y", labelcolor=_COLOR_ACC)

    # ── baseline 参考线（removed_count == 0）────────────────────────────────
    ax_lat.axvline(x=0, color="gray", linewidth=1.0, linestyle=":", alpha=0.7)

    # ── 图例（合并两轴）──────────────────────────────────────────────────────
    lines = ln_lat + ln_acc
    labels = [line.get_label() for line in lines]
    # ── 图例（合并两轴）──────────────────────────────────────────────────────
    lines = ln_lat + ln_acc
    labels = [line.get_label() for line in lines]
    ax_lat.legend(
        lines,
        labels,
        loc="upper right",
        # 1. 边框与背景设置
        facecolor="white",  # 白色背景
        framealpha=0,  # 透明度
        borderpad=0.2,  # 图例整体的内边距（边框到内容的距离，默认通常为 0.4）
        labelspacing=0.3,  # 图例项之间的垂直行间距（默认通常为 0.5）
        handletextpad=0.5,  # 图例标识（线/点）与对应文字之间的水平间距（默认通常为 0.8）
        borderaxespad=0.2,  # 图例的外边距（图例边框到坐标轴边缘的距离，默认通常为 0.5）
    )

    # ── 标题 ──────────────────────────────────────────────────────────────────
    title = "Inference Latency vs. Accuracy Trade-off"
    ax_lat.set_title(title)

    # ── 保存 ──────────────────────────────────────────────────────────────────
    save_fig_for_ieee(save_path, fig)
    plt.close(fig)


def plot_from_csv(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # 类型转换
    for r in rows:
        r["removed_count"] = int(r["removed_count"])
        r["latency_ms"] = float(r["latency_ms"])
        r["acc_top1"] = float(r["acc_top1"])
    rows.sort(key=lambda r: r["removed_count"])

    save_path = csv_path.with_name(csv_path.stem + "_plot")
    plot_inference_tradeoff(rows, save_path)


if __name__ == "__main__":
    example_csv = Path(
        "/root/autodl-tmp/ResidualRemove/Results/Exp1_Motivation/Motivation1_Inference_cost/20260427_1132_tradeoff.csv"
    )
    plot_from_csv(example_csv)
