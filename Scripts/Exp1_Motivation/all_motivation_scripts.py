"""Scripts/Exp1_Motivation/all_motivation_scripts.py
一键运行所有动机实验（顺序执行，共享同一套参数）
"""

import argparse

from Scripts.Exp1_Motivation.run1_inference_tradeoff import (
    build_parser as build_parser_1,
)
from Scripts.Exp1_Motivation.run1_inference_tradeoff import (
    main as run1,
)
from Scripts.Exp1_Motivation.run2_collaborate_speedup import (
    build_parser as build_parser_2,
)
from Scripts.Exp1_Motivation.run2_collaborate_speedup import (
    main as run2,
)
from Scripts.Exp1_Motivation.run3_residual_features import (
    build_parser as build_parser_3,
)
from Scripts.Exp1_Motivation.run3_residual_features import (
    main as run3,
)
from Scripts.Utils.script_common import add_common_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="一键运行所有动机实验（Exp1: run1 + run2 + run3）"
    )
    add_common_args(parser)
    # run1 专属参数
    parser.add_argument("--latency-reps", type=int, default=20)
    parser.add_argument("--latency-warmup", type=int, default=5)
    # run2 专属参数
    parser.add_argument("--serialize-reps", type=int, default=20)
    # 跳过控制
    parser.add_argument(
        "--skip",
        nargs="*",
        type=int,
        choices=[1, 2, 3],
        default=[],
        metavar="N",
        help="跳过指定编号的实验，例如 --skip 2 3",
    )
    return parser


def _fill_args(
    base_args: argparse.Namespace, sub_parser: argparse.ArgumentParser
) -> argparse.Namespace:
    """用 sub_parser 的默认值填充缺失字段，再把 base_args 的值覆盖进去。"""
    sub_defaults = vars(sub_parser.parse_args([]))
    merged = {**sub_defaults, **vars(base_args)}
    return argparse.Namespace(**merged)


def main(args: argparse.Namespace) -> None:
    skip = set(args.skip or [])
    experiments = [
        (1, "推理延迟 & 精度 tradeoff", build_parser_1, run1),
        (2, "协同推理传输开销", build_parser_2, run2),
        (3, "残差分支 L2-norm & 余弦相似度", build_parser_3, run3),
    ]

    total = len(experiments) - len(skip)
    done = 0
    for idx, desc, build_parser_fn, run_fn in experiments:
        if idx in skip:
            print(f"\n{'=' * 60}")
            print(f"[Exp1-All] ⏭  跳过实验 {idx}：{desc}")
            continue

        done += 1
        print(f"\n{'=' * 60}")
        print(f"[Exp1-All] ▶  [{done}/{total}] 实验 {idx}：{desc}")
        print(f"{'=' * 60}")

        sub_args = _fill_args(args, build_parser_fn())
        run_fn(sub_args)
        print(f"[Exp1-All] ✓  实验 {idx} 完成")

    print(f"\n{'=' * 60}")
    print(f"[Exp1-All] 全部完成（{done}/{total} 个实验）")


if __name__ == "__main__":
    # nohup python Scripts/Exp1_Motivation/all_motivation_scripts.py \
    #     --model resnet50 --dataset imagenet100 --batch-size 128 \
    #     > Results/Exp1_Motivation/all_$(date +%Y%m%d_%H%M).log 2>&1 &
    parser = build_parser()
    args = parser.parse_args()
    args.batch_size = 128
    main(args)
