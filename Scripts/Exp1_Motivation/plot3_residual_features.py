"""Scripts/Exp1_Motivation/plot_residual_stats.py"""

"""
Scripts/Exp1_Motivation/plot_residual_stats.py

绘制动机实验4（残差分支可行性分析）的可视化图：
  横轴：残差块索引（网络深度）
  左Y轴（蓝）：L2-norm 比值 ‖F(x)‖ / ‖x‖，阴影表示 ±std
  右Y轴（红）：余弦相似度 cosine(F(x), x)，阴影表示 ±std

论点：
  - L2-ratio 不极大 → identity 分支量级与主路相当，不可忽略
  - cosine 不接近 1  → x 携带独立方向信息，直接删除会丢失有意义的信号

用法（单独运行）：
    python -m Scripts.Exp1_Motivation.plot_residual_stats
"""

import csv
from pathlib import Path
from datetime import datetime
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

from Src.Plots.plot_utils import set_ieee_style, save_fig_for_ieee
from Src.paras import COLORS

_CLR_RATIO  = COLORS["blue"]
_CLR_COSINE = COLORS["red"]
_CLR_GRID   = COLORS["grey"]


# ─────────────────────────────────────────────────────────────────────────────
def load_rows_from_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_residual_stats(
    rows: list[dict],
    save_dir: Path | None = None,
    model_name: str | None = None,
) -> None:
    """
    绘制 L2-norm 比值与余弦相似度的逐块分布图（双 Y 轴 + 标准差阴影）。

    Parameters
    ----------
    rows : list[dict]
        来自 run4_residual_stats.py 的 CSV 行，每行对应一个残差块。
    save_dir : Path, optional
        图片保存目录。None 时仅展示。
    model_name : str, optional
        用于标题和文件名。
    """
    # ── 1. 解析数据（按 block_idx 排序，保证网络深度顺序）────────────────────
    rows_sorted = sorted(rows, key=lambda r: int(r["block_idx"]))

    x           = np.array([int(r["block_idx"])       for r in rows_sorted])
    ratio_mean  = np.array([float(r["l2_ratio_mean"]) for r in rows_sorted])
    ratio_std   = np.array([float(r["l2_ratio_std"])  for r in rows_sorted])
    cos_mean    = np.array([float(r["cosine_mean"])   for r in rows_sorted])
    cos_std     = np.array([float(r["cosine_std"])    for r in rows_sorted])

    # ── 2. 画布（双 Y 轴）────────────────────────────────────────────────────
    set_ieee_style(mode="single")

    fig, ax_ratio = plt.subplots()
    ax_cos = ax_ratio.twinx()   # 共享 X 轴，独立右 Y 轴

    # ── 3. L2-norm 比值（左轴，蓝色）────────────────────────────────────────
    ax_ratio.plot(
        x, ratio_mean,
        color=_CLR_RATIO, marker="o", markersize=3.5,
        linewidth=2, label=r"$\|F(x)\| / \|x\|$", zorder=3,
    )
    ax_ratio.fill_between(
        x,
        ratio_mean - ratio_std,
        ratio_mean + ratio_std,
        color=_CLR_RATIO, alpha=0.15, zorder=2,
    )
    # 参考线：ratio = 1 表示两路幅度相当
    ax_ratio.axhline(
        y=1.0, color=_CLR_RATIO, linestyle=":", linewidth=1.0, alpha=0.5, zorder=1,
    )

    ax_ratio.set_xlabel("Number of Removed Residual Blocks")
    ax_ratio.set_ylabel(r"L2-Norm Ratio  $\|F(x)\| / \|x\|$", color=_CLR_RATIO)
    ax_ratio.tick_params(axis="y", labelcolor=_CLR_RATIO)
    ax_ratio.set_xlim(left=x[0], right=x[-1]+1)
    ax_ratio.xaxis.set_major_locator(MaxNLocator(integer=True))
    ratio_min = np.min(ratio_mean - ratio_std)
    ratio_max = np.max(ratio_mean + ratio_std)
    ratio_range = ratio_max - ratio_min if ratio_max != ratio_min else 1.0
    ax_ratio.set_ylim(bottom=ratio_min - 0.1 * ratio_range, top=ratio_max + 0.1 * ratio_range)

    # ── 4. 余弦相似度（右轴，红色）──────────────────────────────────────────
    ax_cos.plot(
        x, cos_mean,
        color=_CLR_COSINE, marker="s", markersize=3.5,
        linewidth=2, linestyle="--", label=r"$\mathrm{cos}(F(x),\, x)$", zorder=3,
    )
    ax_cos.fill_between(
        x,
        cos_mean - cos_std,
        cos_mean + cos_std,
        color=_CLR_COSINE, alpha=0.12, zorder=2,
    )
    # 参考线：cosine = 0 表示正交（x 携带完全独立信息）
    ax_cos.axhline(
        y=0.0, color=_CLR_COSINE, linestyle=":", linewidth=1.0, alpha=0.5, zorder=1,
    )

    ax_cos.set_ylabel(r"Cosine Similarity  $\cos(F(x),\, x)$", color=_CLR_COSINE)
    ax_cos.tick_params(axis="y", labelcolor=_CLR_COSINE)
  
    cos_min = np.min(cos_mean - cos_std)
    cos_max = np.max(cos_mean + cos_std)
    cos_range = cos_max - cos_min if cos_max != cos_min else 1.0
    ax_cos.set_ylim(bottom=cos_min - 0.1 * cos_range, top=cos_max + 0.1 * cos_range)

    # ── 5. 合并图例（两个轴各自的 handle 合并到一个 legend）────────────────
    handles_r, labels_r = ax_ratio.get_legend_handles_labels()
    handles_c, labels_c = ax_cos.get_legend_handles_labels()
    ax_ratio.legend(
        handles_r + handles_c, labels_r + labels_c,
        loc="upper left", fontsize=9,
        frameon=True, handlelength=1.5,
        handletextpad=0.3, borderpad=0.5,
    )

    # ── 6. 网格（只画在 ratio 轴上，避免双轴网格重叠混乱）──────────────────
    ax_ratio.grid(axis="both", linestyle="--", alpha=0.35)
    ax_ratio.set_axisbelow(True)
    ax_cos.set_axisbelow(True)

    title_model = model_name or (rows[0].get("model", "") if rows else "")
    if title_model:
        ax_ratio.set_title(
            f"Residual Branch Analysis",
            fontsize=11,
        )

    plt.tight_layout(pad=0.4)

    # ── 7. 保存 ──────────────────────────────────────────────────────────────
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{title_model}" if title_model else "ResidualStats"
        save_fig_for_ieee(save_dir / stem, fig=fig)

    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    save_dir    = Path("/home/ResidualRemove/Results/Exp1_Motivation/Motivation4_Residual_stats")
    csv_path = save_dir / "20260412_2157_residual_stats.csv"

    rows       = load_rows_from_csv(csv_path)
    model_name = rows[0]["model"] if rows else None

    csv_stem = Path(csv_path).stem
    plot_residual_stats(rows, save_dir=save_dir, model_name=csv_stem)