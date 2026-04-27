"""Scripts/Exp1_Motivation/run4_residual_stats.py
动机实验4：残差分支 F(x) 和identity分支 x 的 L2-norm 比值与余弦相似度。
  论点：identity 分支（即 x）是结构的信号，不是可以忽略的噪声，证明"为什么不能直接删残差"
  (1) L2-norm 比值 = ‖F(x)‖ / ‖x‖
      如果 比值接近 1，残差分支的幅度和主路输出相当，不可忽略
      如果 比值远大于 1，主路输出主导，残差相对较小
      期望：比值合理（不是极大值），证明 x 的量级是有实质意义的

  (2) 余弦相似度 = cosine(F(x), x)
      如果 接近 0，说明两者方向正交，x 携带了 F(x) 没有的信息
      如果 接近 1，说明 x 和 F(x) 高度相关，残差几乎是冗余的
      期望：余弦相似度不接近 1（即 x 携带独立信息），证明直接扔掉 x 会丢失有意义的方向信息
"""

import argparse
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from Configs.paras import RESULT_DIR_1
from Scripts.Utils.script_common import add_common_args, build_setup
from Src.Utils.runtime import write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="每个残差块的 L2-norm 比值与余弦相似度（为什么不能直接删残差）"
    )
    add_common_args(parser)
    parser.add_argument(
        "--output",
        default=None,
        help="输出 CSV 的路径（默认 Results/Exp1_Motivation/Motivation4_Residual_stats/time_residual_stats.csv）",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup = build_setup(args, compensator_name="identity")

    model = setup["model"]
    bundle = setup["bundle"]
    device = setup["device"]
    blocks = model.get_block_names()

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = Path(
        args.output
        or RESULT_DIR_1
        / "Exp1_Motivation"
        / "Motivation4_Residual_stats"
        / f"{current_time}_residual_stats.csv"
    )

    print("\n[Exp1-Stats] 开始残差统计实验")
    print(f"[Exp1-Stats] 共 {len(blocks)} 个残差块")
    print(f"[Exp1-Stats] 结果将写入：{output_path}\n")

    # ── 累加器设计 ────────────────────────────────────────────────────────────
    # 对每个 block，我们跨多个 batch 累计以下数值，最后除以 batch 数得到均值：
    #   ratio_sum   : L2-norm 比值的逐 batch 均值之和
    #   ratio_sq_sum: 比值平方的逐 batch 均值之和（用于计算标准差）
    #   cos_sum     : 余弦相似度的逐 batch 均值之和
    #   cos_sq_sum  : 余弦相似度平方的逐 batch 均值之和
    #   count       : 累计的 batch 数量
    #
    # 注意：这里累计的是"batch 内均值"，而不是"所有样本的原始值"。
    # 这是一种近似，但在 batch size 稳定的情况下结果和直接累计样本是一致的。
    acc = defaultdict(
        lambda: {
            "ratio_sum": 0.0,
            "ratio_sq_sum": 0.0,
            "cos_sum": 0.0,
            "cos_sq_sum": 0.0,
            "count": 0,
        }
    )

    # ── 主循环：遍历验证集，逐 batch 收集统计量 ──────────────────────────────
    model.eval()
    total_batches = len(bundle.val_loader)
    print_interval = max(1, total_batches // 10)  # 每完成 10% 打印一次进度

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(bundle.val_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            images = images.to(device)

            # return_residual_stats=True 让每个 PatchedBlock 在 forward_collect
            # 里同时返回 plain（F(x)）和 identity（x）两个中间张量
            output = model(images, mode="full", return_residual_stats=True)

            for block_name, block_stats in output["residual_stats"].items():
                # flatten(1) 把 (B, C, H, W) 压成 (B, C*H*W)，
                # 这样 norm(dim=1) 计算的是每个样本的整体 L2 范数，
                # cosine_similarity(dim=1) 计算的是每个样本的方向相似度
                plain = block_stats["plain"].flatten(1)  # F(x)，主路输出
                identity = block_stats["identity"].flatten(1)  # x，残差分支

                # L2-norm 比值：‖F(x)‖ / ‖x‖
                # 分母加 1e-8 防止 identity 全零时除零（理论上不会发生，但防御性编程）
                ratio = plain.norm(dim=1) / (identity.norm(dim=1) + 1e-8)
                cosine = F.cosine_similarity(plain, identity, dim=1)

                # 取 batch 内均值，累加到各自的 sum 和 sq_sum
                ratio_mean = float(ratio.mean().item())
                ratio_sq = float((ratio**2).mean().item())
                cos_mean = float(cosine.mean().item())
                cos_sq = float((cosine**2).mean().item())

                acc[block_name]["ratio_sum"] += ratio_mean
                acc[block_name]["ratio_sq_sum"] += ratio_sq
                acc[block_name]["cos_sum"] += cos_mean
                acc[block_name]["cos_sq_sum"] += cos_sq
                acc[block_name]["count"] += 1

            # 进度打印：不要每个 batch 都打，太刷屏；每 10% 打一次
            if (batch_idx + 1) % print_interval == 0 or batch_idx == 0:
                done = batch_idx + 1
                limit = args.max_batches or total_batches
                print(f"  [{done}/{limit}] batch 处理中...")

    # ── 汇总统计量，计算均值和标准差 ─────────────────────────────────────────
    print("\n[Exp1-Stats] 所有 batch 处理完成，开始汇总...\n")

    rows: list[dict] = []

    # block_order 是模型中块的实际顺序，用它来排序确保 CSV 行顺序和网络深度一致
    for block_idx, block_name in enumerate(blocks):
        if block_name not in acc:
            # 某些块可能没有参与统计（比如 MobileNet 里没有残差连接的块）
            continue

        a = acc[block_name]
        n = max(a["count"], 1)

        # 均值
        ratio_mean = a["ratio_sum"] / n
        cos_mean = a["cos_sum"] / n

        # 标准差：Var(X) = E[X²] - E[X]²，std = sqrt(Var)
        # 这里计算的是 batch 间均值的标准差，反映统计量在不同 batch 间的稳定性
        # 低标准差 = 信号稳定 = 残差是有规律的结构性信号，而不是随机噪声
        ratio_var = max(a["ratio_sq_sum"] / n - ratio_mean**2, 0.0)
        cos_var = max(a["cos_sq_sum"] / n - cos_mean**2, 0.0)
        ratio_std = math.sqrt(ratio_var)
        cos_std = math.sqrt(cos_var)

        print(
            f"  {block_name:20s}  "
            f"L2-ratio={ratio_mean:.4f}±{ratio_std:.4f}  "
            f"cosine={cos_mean:.4f}±{cos_std:.4f}"
        )

        rows.append(
            {
                "dataset": bundle.source,
                "model": args.model,
                # block_idx 从 0 开始，方便后续画图时作为 X 轴（代表网络深度）
                "block_idx": block_idx,
                "block": block_name,
                "num_batches": n,
                # L2-norm 比值统计
                "l2_ratio_mean": round(ratio_mean, 6),
                "l2_ratio_std": round(ratio_std, 6),
                # 余弦相似度统计
                "cosine_mean": round(cos_mean, 6),
                "cosine_std": round(cos_std, 6),
            }
        )

    saved = write_csv(output_path, rows)
    print(f"\n[Exp1-Stats] 完成。结果已保存至：{saved}")
    print(f"[Exp1-Stats] 共 {len(rows)} 行记录，对应 {len(rows)} 个有效残差块。")


if __name__ == "__main__":
    main()
