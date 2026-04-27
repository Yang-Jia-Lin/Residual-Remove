"""Scripts/Exp1_Motivation/run1_inference_cost.py
动机实验1：删除残差的推理收益和损失分析
在同样的【模型/硬件/数据】下，逐阶段删除残差后的【精度/FLOPS/延迟】变化
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch

from Configs.paras import RESULT_DIR_1
from Scripts.Utils.script_common import add_common_args, build_setup
from Src.Metrics.latency import measure_latency
from Src.Metrics.memory import measure_activation_memory
from Src.Utils.runtime import write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="量化删除残差后的推理收益和损失：推理延迟 + 精度"
    )
    add_common_args(parser)
    parser.add_argument(
        "--latency-reps",
        type=int,
        default=20,
        help="延迟测量的重复次数，取均值（默认 20）",
    )
    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=5,
        help="延迟测量的预热次数，预热结果不计入统计（默认 5）",
    )
    return parser


def main(args) -> None:
    # 初始化环境
    setup = build_setup(args, compensator_name="identity")
    model = setup["model"]
    bundle = setup["bundle"]
    device = setup["device"]
    blocks = model.get_block_names()
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = Path(
        args.output
        if getattr(args, "output", None)
        else RESULT_DIR_1
        / "Motivation1_Inference_cost"
        / f"{current_time}_trade-off.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    measure_images, _ = next(iter(bundle.val_loader))  # 取第一个 batch 作为固定测量输入
    measure_images = measure_images.to(device)
    print(f"[Exp1-InferenceCost] 所有结果均使用同一 batch：{measure_images.shape}\n")
    if args.memory_limit_gb is not None and device.type == "cuda":
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        fraction = args.memory_limit_gb / total_gb
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        print(
            f"[Exp1] 显存限制：{args.memory_limit_gb:.1f} GB / {total_gb:.1f} GB（{fraction:.1%}）"
        )

    # ── 第 0 步：full mode
    rows: list[dict] = []
    fwd_kwargs = {"mode": "full", "removed_blocks": None}
    act_result = measure_activation_memory(model, measure_images, **fwd_kwargs)
    lat_result = measure_latency(
        model,
        measure_images,
        repetitions=args.latency_reps,
        warmup=args.latency_warmup,
        **fwd_kwargs,
    )
    full_act_mb = act_result.activation_peak_mb
    full_latency_ms = lat_result.mean_ms
    print(f"[0/{len(blocks)}] 测量 baseline（full mode，保留所有残差）...")
    print(
        f"\t激活内存峰值：{'N/A' if full_act_mb < 0 else f'{full_act_mb:.2f} MB'}\n"
        f"\t延迟：{full_latency_ms:.3f} ms"
    )
    rows.append(
        {
            "dataset": bundle.source,
            "model": args.model,
            "batch_size": measure_images.size(0),
            "mode": "full",
            "removed_count": 0,
            "removed_blocks": "",
            "activation_peak_mb": round(full_act_mb, 2) if full_act_mb >= 0 else -1,
            "saved_activation_mb": 0.0,
            "saved_activation_pct": 0.0,
            "latency_ms": round(full_latency_ms, 4),
            "saved_latency_ms": 0.0,
            "speedup": 1.0,
        }
    )

    # ── 第 1~N 步：逐步删除
    for remove_count in range(1, len(blocks) + 1):
        removed = blocks[-remove_count:]
        fwd_kwargs = {"mode": "plain", "removed_blocks": removed}
        print(
            f"[{remove_count}/{len(blocks)}] 删除最后 {remove_count} 个块"
            f"（{removed[0]} → {removed[-1]}）..."
        )

        # 激活内存
        act_result = measure_activation_memory(model, measure_images, **fwd_kwargs)
        act_mb = act_result.activation_peak_mb
        saved_act_mb = (
            full_act_mb - act_mb if full_act_mb >= 0 and act_mb >= 0 else -1.0
        )
        saved_act_pct = (
            saved_act_mb / full_act_mb * 100
            if full_act_mb > 0 and saved_act_mb >= 0
            else -1.0
        )
        act_str = (
            "N/A" if act_mb < 0 else f"{act_mb:.2f} MB (节省 {saved_act_mb:.2f} MB)"
        )

        # 延迟
        lat_result = measure_latency(
            model,
            measure_images,
            repetitions=args.latency_reps,
            warmup=args.latency_warmup,
            **fwd_kwargs,
        )
        latency_ms = lat_result.mean_ms
        saved_lat_ms = full_latency_ms - latency_ms
        speedup = full_latency_ms / latency_ms if latency_ms > 0 else 1.0

        print(
            f"\t激活内存：{act_str}\n"
            f"\t延迟：{latency_ms:.3f} ms  (加速：{speedup:.3f}×)"
        )
        rows.append(
            {
                "dataset": bundle.source,
                "model": args.model,
                "batch_size": measure_images.size(0),
                "mode": "plain",
                "removed_count": remove_count,
                "removed_blocks": ",".join(removed),
                "activation_peak_mb": round(act_mb, 2) if act_mb >= 0 else -1,
                "saved_activation_mb": round(saved_act_mb, 2)
                if saved_act_mb >= 0
                else -1,
                "saved_activation_pct": round(saved_act_pct, 2)
                if saved_act_pct >= 0
                else -1,
                "latency_ms": round(latency_ms, 4),
                "saved_latency_ms": round(saved_lat_ms, 4),
                "speedup": round(speedup, 4),
            }
        )
    saved = write_csv(output_path, rows)
    print(f"\n[Exp1-InferenceCost] 完成。结果已保存至：{saved}")

    # 打印简洁摘要
    if len(rows) > 1:
        last = rows[-1]
        print("\n[Exp1-InferenceCost] ── 摘要 ──")
        if last["activation_peak_mb"] >= 0:
            print(
                f"  激活内存：{rows[0]['activation_peak_mb']:.2f} MB → {last['activation_peak_mb']:.2f} MB"
                f"（节省 {last['saved_activation_pct']:.1f}%）"
            )
        print(
            f"  推理延迟：{rows[0]['latency_ms']:.3f} ms → {last['latency_ms']:.3f} ms"
            f"（加速比 {last['speedup']:.3f}×）"
        )


if __name__ == "__main__":
    # nohup Scripts/Exp1_Motivation/run1_inference_cost.sh < /dev/null > Results/Exp1_Motivation/Motivation1_Inference_cost/$(date +%Y%m%d_%H%M).log 2>&1 &
    args = build_parser().parse_args()
    args.batch_size = 128
    # args.memory_limit_gb = 8  # 可选：限制显存使用，模拟在显存受限环境下的效果
    main(args)
