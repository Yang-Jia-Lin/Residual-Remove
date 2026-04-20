"""
Scripts/Exp1_Motivation/plot_combined_cost.py

将峰值显存、推理延迟、Top-1 精度三条曲线绘制在同一张图中。
横轴：移除的残差块数目
纵轴：相对 baseline 的百分比（baseline = 100%）
  - 内存 / 延迟向下 → 代表收益
  - 精度向下         → 代表代价

用法（单独运行）：
    python -m Scripts.Exp1_Motivation.plot_combined_cost
"""

import csv
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from Src.Plots.plot_utils import set_ieee_style, save_fig_for_ieee
from Src.paras import COLORS

_CLR_MEMORY  = COLORS["blue"]
_CLR_LATENCY = COLORS["red"]
_CLR_ACC     = COLORS["green"]
_CLR_BASE    = COLORS["grey"]


# ─────────────────────────────────────────────────────────────────────────────
def load_rows_from_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _extract_plain(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, float]:
    """
    返回 (x_counts, y_values_pct_of_baseline, baseline)
    baseline 取 mode=="full" 行的对应字段值，y 归一化为 baseline 的百分比。
    """
    baseline_row = next((r for r in rows if r["mode"] == "full"), None)
    if baseline_row is None:
        raise ValueError("CSV 中缺少 mode=='full' 的 baseline 行")
    base_val = float(baseline_row[key])

    plain = sorted(
        [r for r in rows if r["mode"] == "plain"],
        key=lambda r: int(r["removed_count"]),
    )
    counts = np.array([int(r["removed_count"]) for r in plain])
    vals   = np.array([float(r[key])           for r in plain])
    pct    = vals / base_val * 100.0
    return counts, pct, base_val


# ─────────────────────────────────────────────────────────────────────────────
def plot_combined_cost(
    cost_rows: list[dict],   # 来自 run1_inference_cost.py 的 CSV
    acc_rows:  list[dict],   # 来自 run3_acc_drop.py 的 CSV
    save_dir:  Path | None = None,
    model_name: str | None = None,
) -> None:
    """
    将峰值显存、推理延迟、Top-1 精度绘制在同一坐标系内（归一化为 baseline %）。
    """
    # ── 1. 提取各条曲线 ──────────────────────────────────────────────────────
    has_memory = True
    try:
        x_mem,  y_mem,  base_mem  = _extract_plain(cost_rows, "peak_mb")
        if base_mem < 0:          # CPU 运行时无显存数据
            has_memory = False
    except Exception:
        has_memory = False

    x_lat,  y_lat,  base_lat  = _extract_plain(cost_rows, "latency_ms")
    x_acc,  y_acc,  base_acc  = _extract_plain(acc_rows,  "top1")

    # ── 2. 画布 ──────────────────────────────────────────────────────────────
    set_ieee_style(mode="single")

    fig, ax = plt.subplots()

    # baseline 参考线（100%）
    x_max = int(max(x_lat.max(), x_acc.max())) + 1
    ax.axhline(y=100, color=_CLR_BASE if True else "grey",
               linestyle="--", linewidth=1.0, alpha=0.6, zorder=1)

    # ── 3. 三条曲线 ──────────────────────────────────────────────────────────
    if has_memory:
        ax.plot(x_mem, y_mem,
                color=_CLR_MEMORY, marker="o", markersize=4,
                linewidth=2, label="Peak Memory", zorder=3)

    ax.plot(x_lat, y_lat,
            color=_CLR_LATENCY, marker="s", markersize=4,
            linewidth=2, label="Latency", zorder=3)

    # ax.plot(x_acc, y_acc,
    #         color=_CLR_ACC, marker="^", markersize=4,
    #         linewidth=2, label="Top-1 Accuracy", zorder=3)

    # ── 4. 末端标注（绝对值，方便读图）────────────────────────────────────────
    def _annotate_end(ax, x_arr, y_pct_arr, base_val, unit, color, offset=(6, 0)):
        abs_val = y_pct_arr[-1] / 100.0 * base_val
        ax.annotate(
            f"{abs_val:.1f}{unit}",
            xy=(x_arr[-1], y_pct_arr[-1]),
            xytext=offset, textcoords="offset points",
            fontsize=8, color=color, va="center",
        )

    if has_memory:
        _annotate_end(ax, x_mem, y_mem, base_mem, " MB", _CLR_MEMORY)
    _annotate_end(ax, x_lat, y_lat, base_lat, " ms", _CLR_LATENCY)
    # _annotate_end(ax, x_acc, y_acc, base_acc, "%",   _CLR_ACC)

    # ── 5. 轴标签、图例、网格 ────────────────────────────────────────────────
    ax.set_xlabel("Number of Removed Residual Blocks")
    ax.set_ylabel("Relative to Baseline (%)")
    ax.set_xlim(left=0, right=x_max)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(
        loc="lower left", fontsize=9,
        frameon=True, handlelength=1.5,
        handletextpad=0.3, borderpad=0.5,
    )
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    title_model = model_name or (cost_rows[0].get("model", "") if cost_rows else "")
    if title_model:
        ax.set_title(f"Residual Removal Trade-off", fontsize=11)

    plt.tight_layout(pad=0.4)

    # ── 6. 保存 ──────────────────────────────────────────────────────────────
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{title_model}" if title_model else "Combined"
        save_fig_for_ieee(save_dir / stem, fig=fig)

    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _ROOT = Path("/root/autodl-tmp/Residual-Remove/Results/Exp1_Motivation")
    save_dir = _ROOT / "Motivation1_Inference_cost"
    cost_csv = _ROOT / "Motivation1_Inference_cost" / "20260412_2150_memory_cost.csv"
    acc_csv  = _ROOT / "Motivation3_Acc_drop"       / "20260412_2157_acc_drop.csv"

    cost_rows  = load_rows_from_csv(cost_csv)
    acc_rows   = load_rows_from_csv(acc_csv)
    model_name = cost_rows[0]["model"] if cost_rows else None
    
    csv_stem = Path(cost_csv).stem 
    plot_combined_cost(cost_rows, acc_rows, save_dir=save_dir, model_name=csv_stem)