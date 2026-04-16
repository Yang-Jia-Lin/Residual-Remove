"""
Scripts/Exp1_Motivation/plot_collab_cost.py

绘制动机实验2（协同推理传输开销）的可视化图：
  横轴：残差块索引（代表模型切分点的位置 / 网络深度）
  纵轴：链式拓扑相对 DAG 拓扑节省的总系统时间百分比（total_saved_pct）
  多条线：不同带宽场景（bandwidth_mbps）

用法（单独运行）：
    python -m Scripts.Exp1_Motivation.plot_collab_cost
"""

import csv
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

from Src.Plots.plot_utils import set_ieee_style, save_fig_for_ieee
from Src.paras import COLORS

# 带宽曲线颜色：取调色板中语义较中性的几个
_BW_COLORS = [COLORS["blue"], COLORS["red"], COLORS["green"], COLORS["purple"]]


# ─────────────────────────────────────────────────────────────────────────────
def load_rows_from_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_collab_cost(
    rows: list[dict],
    save_dir: Path | None = None,
    model_name: str | None = None,
) -> None:
    """
    绘制协同推理传输开销图。

    Parameters
    ----------
    rows : list[dict]
        来自 run2_collab_cost.py 的 CSV 行，每行是 (block × bandwidth) 的一条记录。
    save_dir : Path, optional
        图片保存目录。None 时仅展示。
    model_name : str, optional
        用于标题和文件名。
    """
    # ── 1. 整理数据：按 bandwidth 分组，提取各切分点的节省比 ─────────────────
    # block 顺序：CSV 写入时已按网络深度顺序排列，取第一个 bandwidth 的行序即可
    bandwidths = sorted(set(float(r["bandwidth_mbps"]) for r in rows))
    # 所有 block 名称（按出现顺序，保留网络深度顺序）
    seen = set()
    block_names = []
    for r in rows:
        if r["block"] not in seen:
            seen.add(r["block"])
            block_names.append(r["block"])

    x = np.arange(len(block_names))
    block_to_idx = {name: i for i, name in enumerate(block_names)}

    # ── 2. 画布 ──────────────────────────────────────────────────────────────
    set_ieee_style(mode="single")

    fig, ax = plt.subplots()

    # ── 3. 每种带宽画一条线 ──────────────────────────────────────────────────
    markers = ["o", "s", "^", "D"]
    all_y = []  # 【新增】用于收集所有的Y值，以便计算上下界
    for bw_idx, bw in enumerate(bandwidths):
        bw_rows = sorted(
            [r for r in rows if float(r["bandwidth_mbps"]) == bw],
            key=lambda r: block_to_idx[r["block"]],
        )
        y = np.array([float(r["total_saved_pct"]) for r in bw_rows])
        xi = np.array([block_to_idx[r["block"]] + 1 for r in bw_rows])

        all_y.extend(y)  # 【新增】将当前带宽的Y值加入集合

        color  = _BW_COLORS[bw_idx % len(_BW_COLORS)]
        marker = markers[bw_idx % len(markers)]
        label  = f"{int(bw)} Mbps" if bw == int(bw) else f"{bw} Mbps"

        ax.plot(
            xi, y,
            color=color, marker=marker, markersize=4,
            linewidth=2, label=label, zorder=3,
        )
    if all_y:
        y_min, y_max = np.min(all_y), np.max(all_y)
        y_range = y_max - y_min if y_max != y_min else 1.0
        ax.set_ylim(bottom=y_min - 0.1 * y_range, top=y_max + 0.1 * y_range)

    # ── 4. 参考线：节省 0%（DAG = Chain）────────────────────────────────────
    ax.axhline(y=0, color=COLORS["grey"], linestyle="--", linewidth=1.0,
               alpha=0.6, zorder=1)

    # ── 5. X 轴刻度：只标首/尾及均匀间隔，避免过密 ──────────────────────────
    n_blocks = len(block_names)
    
    ax.set_xlabel("Number of Removed Residual Blocks")
    ax.set_ylabel("Transmission Cost Saved (%)\n(Chain vs. DAG)")
    ax.set_xlim(left=0, right=n_blocks+1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(
        title="Bandwidth", title_fontsize=9,
        fontsize=9, loc="best",
        frameon=True, handlelength=1.5,
        handletextpad=0.3, borderpad=0.5,
    )
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    title_model = model_name or (rows[0].get("model", "") if rows else "")
    if title_model:
        ax.set_title(
            f"Collaborative Inference Transmission Cost",
            fontsize=11,
        )

    plt.tight_layout(pad=0.4)

    # ── 6. 保存 ──────────────────────────────────────────────────────────────
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{title_model}" if title_model else "CollabCost"
        save_fig_for_ieee(save_dir / stem, fig=fig)

    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    save_dir    = Path("/home/ResidualRemove/Results/Exp1_Motivation/Motivation2_Collaborate_cost")
    csv_path = save_dir / "20260412_2157_system_cost.csv"

    rows       = load_rows_from_csv(csv_path)
    model_name = rows[0]["model"] if rows else None

    csv_stem = Path(csv_path).stem 
    plot_collab_cost(rows, save_dir=save_dir, model_name=csv_stem)