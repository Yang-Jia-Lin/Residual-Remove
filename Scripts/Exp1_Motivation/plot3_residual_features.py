"""Scripts/Exp1_Motivation/plot3_residual_features.py"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from Configs.paras import COLORS
from Src.Utils.plot_utils import save_fig_for_ieee, set_ieee_style

_CLR_RATIO = COLORS["blue"]
_CLR_COSINE = COLORS["red"]


def plot_residual_stats(rows: list[dict], save_path: Path) -> None:
    # ── 解析数据 ──────────────────────────────────────────────────────────────
    rows_sorted = sorted(rows, key=lambda r: int(r["block_idx"]))

    x = np.array([int(r["block_idx"]) for r in rows_sorted])
    ratio_mean = np.array([float(r["l2_ratio_mean"]) for r in rows_sorted])
    ratio_std = np.array([float(r["l2_ratio_std"]) for r in rows_sorted])
    cos_mean = np.array([float(r["cosine_mean"]) for r in rows_sorted])
    cos_std = np.array([float(r["cosine_std"]) for r in rows_sorted])

    # ── 画布 ──────────────────────────────────────────────────────────────────
    set_ieee_style(mode="single")
    fig, ax_ratio = plt.subplots()
    ax_cos = ax_ratio.twinx()

    # ── 左轴：L2-norm 比值 ────────────────────────────────────────────────────
    ln_ratio = ax_ratio.plot(
        x,
        ratio_mean,
        color=_CLR_RATIO,
        marker="o",
        markersize=3.5,
        label=r"$\|F(x)\| / \|x\|$",
        zorder=3,
    )
    ax_ratio.fill_between(
        x,
        ratio_mean - ratio_std,
        ratio_mean + ratio_std,
        color=_CLR_RATIO,
        alpha=0.15,
        zorder=2,
    )
    ax_ratio.axhline(
        y=1.0, color=_CLR_RATIO, linestyle=":", linewidth=1.0, alpha=0.5, zorder=1
    )

    ax_ratio.set_xlabel("Number of Removed Residual Blocks")
    ax_ratio.set_ylabel(r"L2-Norm Ratio  $\|F(x)\| / \|x\|$", color=_CLR_RATIO)
    ax_ratio.tick_params(axis="y", labelcolor=_CLR_RATIO)
    ax_ratio.set_xlim(left=x[0], right=x[-1] + 1)
    ax_ratio.xaxis.set_major_locator(MaxNLocator(integer=True))

    ratio_min, ratio_max = (
        np.min(ratio_mean - ratio_std),
        np.max(ratio_mean + ratio_std),
    )
    ratio_range = ratio_max - ratio_min if ratio_max != ratio_min else 1.0
    ax_ratio.set_ylim(
        bottom=ratio_min - 0.1 * ratio_range, top=ratio_max + 0.1 * ratio_range
    )

    # ── 右轴：余弦相似度 ──────────────────────────────────────────────────────
    ln_cos = ax_cos.plot(
        x,
        cos_mean,
        color=_CLR_COSINE,
        marker="s",
        markersize=3.5,
        linestyle="--",
        label=r"$\mathrm{cos}(F(x),\, x)$",
        zorder=3,
    )
    ax_cos.fill_between(
        x,
        cos_mean - cos_std,
        cos_mean + cos_std,
        color=_CLR_COSINE,
        alpha=0.12,
        zorder=2,
    )
    ax_cos.axhline(
        y=0.0, color=_CLR_COSINE, linestyle=":", linewidth=1.0, alpha=0.5, zorder=1
    )

    ax_cos.set_ylabel(r"Cosine Similarity  $\cos(F(x),\, x)$", color=_CLR_COSINE)
    ax_cos.tick_params(axis="y", labelcolor=_CLR_COSINE)

    cos_min, cos_max = np.min(cos_mean - cos_std), np.max(cos_mean + cos_std)
    cos_range = cos_max - cos_min if cos_max != cos_min else 1.0
    ax_cos.set_ylim(bottom=cos_min - 0.1 * cos_range, top=cos_max + 0.1 * cos_range)

    # ── 图例（合并两轴）──────────────────────────────────────────────────────
    lines = ln_ratio + ln_cos
    labels = [line.get_label() for line in lines]
    ax_ratio.legend(
        lines,
        labels,
        loc="upper left",
        fontsize=8,
        frameon=True,
        handlelength=1,
        handletextpad=0.2,
        borderpad=0.2,
    )

    # ── 标题 ──────────────────────────────────────────────────────────────────
    ax_ratio.set_title("Residual Branch Analysis")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    save_fig_for_ieee(save_path, fig)
    plt.close(fig)


def plot_from_csv(csv_path: Path) -> None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    save_path = csv_path.with_name(csv_path.stem + "_plot")
    plot_residual_stats(rows, save_path)


if __name__ == "__main__":
    example_csv = Path(
        "/root/autodl-tmp/ResidualRemove/Results/Exp1_Motivation/Motivation3_Residual_feature/20260428_1132_residual_feature.csv"
    )
    plot_from_csv(example_csv)
