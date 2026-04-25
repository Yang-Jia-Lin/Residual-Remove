"""Scripts/Exp1_Motivation/run3_acc_drop.py"""

"""动机实验3：逐步删除残差块的精度下降趋势"""
import argparse
from pathlib import Path
from datetime import datetime

from Configs.paras import RESULT_DIR_1
from Scripts.Utils.script_common import add_common_args, build_setup
from Src.Training_and_Evaluation.evaluator import evaluate_model
from Src.Utils.runtime import write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="逐步删除残差块时模型 Top-1/Top-5 精度的下降趋势"
    )
    add_common_args(parser)
    parser.add_argument(
        "--output", 
        default=None,
        help="输出 CSV 的路径（默认 Results/Exp1_Motivation/Motivation3_Acc_drop/time_acc_drop.csv）"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup = build_setup(args, compensator_name="identity")

    model  = setup["model"]
    bundle = setup["bundle"]
    device = setup["device"]
    blocks = model.get_block_names()

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = Path(args.output or RESULT_DIR_1 / "Motivation3_Acc_drop" / f"{current_time}_acc_drop.csv")

    print(f"\n[Exp1] 开始精度下降实验")
    print(f"[Exp1] 共 {len(blocks)} 个残差块，将依次从最后一个开始删除")
    print(f"[Exp1] 结果将写入：{output_path}\n")

    rows: list[dict] = []

    # ── 第 0 步：full mode 作为 baseline ─────────────────────────────
    print(f"[0/{len(blocks)}] 评估 baseline（full mode，保留所有残差）...")
    full_metrics = evaluate_model(
        model, bundle.val_loader,
        device=device, mode="full",
        max_batches=args.max_batches,
    )
    print(
        f"          loss={full_metrics.loss:.4f}  "
        f"top1={full_metrics.top1:.2f}%  "
        f"top5={full_metrics.top5:.2f}%"
    )

    rows.append({
        "dataset":        bundle.source,
        "model":          args.model,
        "mode":           "full",
        "removed_count":  0,
        "removed_blocks": "",
        "top1":           round(full_metrics.top1, 4),
        "top5":           round(full_metrics.top5, 4),
        "loss":           round(full_metrics.loss, 6),
        "top1_drop":      0.0,
    })

    # ── 第 1~N 步：从最后一个块开始，逐步扩大删除范围 ──────────────────
    # 这个顺序设计是为了让图表上的 X 轴从右到左表示"从深层到浅层"，
    # 和直觉一致：深层块被删对精度影响更大。
    for remove_count in range(1, len(blocks) + 1):
        removed = blocks[-remove_count:]   # 始终包含最后 remove_count 个块
        print(
            f"[{remove_count}/{len(blocks)}] 删除最后 {remove_count} 个块"
            f"（{removed[0]} → {removed[-1]}）..."
        )

        metrics = evaluate_model(
            model, bundle.val_loader,
            device=device, mode="plain",
            removed_blocks=removed,
            max_batches=args.max_batches,
        )

        top1_drop = full_metrics.top1 - metrics.top1
        print(
            f"          loss={metrics.loss:.4f}  "
            f"top1={metrics.top1:.2f}%  "
            f"top5={metrics.top5:.2f}%  "
            f"drop={top1_drop:+.2f}%"
        )

        rows.append({
            "dataset":        bundle.source,
            "model":          args.model,
            "mode":           "plain",
            "removed_count":  remove_count,
            "removed_blocks": ",".join(removed),
            "top1":           round(metrics.top1, 4),
            "top5":           round(metrics.top5, 4),
            "loss":           round(metrics.loss, 6),
            "top1_drop":      round(top1_drop, 4),
        })

    saved = write_csv(output_path, rows)
    print(f"\n[Exp1] 完成。结果已保存至：{saved}")


if __name__ == "__main__":
    main()