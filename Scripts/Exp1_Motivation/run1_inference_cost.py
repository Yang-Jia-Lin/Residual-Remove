"""Scripts/Exp1_Motivation/run1_inference_cost.py"""

"""动机实验1：删除残差的综合收益分析
  (1) 理论 FLOPS 不变
  (2) 实际推理峰值内存下降：
      full mode 在执行 out = F(x) + x 时，内存中同时存在 F(x)、x、out 三个张量。
      plain mode 不保留 x，峰值内存更低，相同显存可以支持更大的 batch 或者同时运行更多的推理实例
  (3) 实际推理时延下降：
      在内存带宽受限的设备上，搬运数据的代价远大于计算本身。
      删除残差减少了内存读写次数，推理速度因此提升
  (4) 精度下降
  
记录在同样的模型、硬件、数据下，逐阶段删除残差后的 精度、FLOPS、内存、延迟 变化
"""
import argparse
import time
import torch
from pathlib import Path
from datetime import datetime

from Scripts.Utils.common import add_common_args, build_setup
from Src.Utils.runtime import write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="量化删除残差后的综合收益：峰值内存 + 推理延迟"
    )
    add_common_args(parser)
    parser.add_argument(
        "--output", 
        default=None,
        help="输出 CSV 路径（默认 Results/Exp1_Motivation/Motivation1_Inference_cost/time_memory_cost.csv）")
    parser.add_argument(
        "--latency-reps", 
        type=int, 
        default=20,
        help="延迟测量的重复次数，取均值（默认 20）")
    parser.add_argument(
        "--latency-warmup", 
        type=int, 
        default=5,
        help="延迟测量的预热次数，预热结果不计入统计（默认 5）")
    return parser


def _measure_one(
    model: torch.nn.Module,
    images: torch.Tensor,
    mode: str,
    removed_blocks: list[str] | None,
    device: torch.device,
    latency_reps: int,
    latency_warmup: int,
) -> tuple[float, float]:
    """对一种配置，同时测量峰值内存和推理延迟，返回 (peak_mb, mean_latency_ms)
    返回：
        peak_mb：GPU 显存峰值（MB）；CPU 上返回 -1.0 表示无法测量。
        mean_latency_ms：多次推理的平均延迟（毫秒）
    """
    model.eval()

    # ── 预热阶段：让 GPU JIT 编译和内存分配稳定下来 ────────────────────
    # 预热的重要性：第一次推理往往触发 CUDA kernel 编译和内存池扩张，
    # 时间远长于稳定状态，不预热会严重高估延迟。
    with torch.no_grad():
        for _ in range(latency_warmup):
            model(images, mode=mode, removed_blocks=removed_blocks)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # ── 峰值内存测量：在一次干净的前向推理里测 ────────────────────────────
    # 预热结束后清零计数器，确保测的是"稳定状态"下的峰值，
    # 而不是包含初始化开销的历史最高值
    # if device.type == "cuda":
    #     torch.cuda.reset_peak_memory_stats(device)

    # with torch.no_grad():
    #     model(images, mode=mode, removed_blocks=removed_blocks)
    # if device.type == "cuda":
    #     torch.cuda.synchronize(device)

    # peak_mb = (
    #     torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    #     if device.type == "cuda"
    #     else -1.0
    # )

    # ── 峰值内存测量：巧妙跳过 Stem 的干扰 ────────────────────────────
    if device.type == "cuda":
        with torch.no_grad():
            # 1. 先把极占显存的 Stem 跑完
            x_stem = model.forward_to_split(images, split_point="stem", mode=mode, removed_blocks=removed_blocks)
            
        # 2. 清除不需要的缓存，并【重置峰值计数器】！
        # 这样接下来的峰值，就是纯粹由“残差块”产生的，不再受 Stem 掩盖
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        with torch.no_grad():
            # 3. 仅测量核心残差阶段和头部的显存峰值
            _ = model.forward_from_split(x_stem, split_point="stem", mode=mode, removed_blocks=removed_blocks)
            
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        # CPU 逻辑保持不变（暂不处理）
        with torch.no_grad():
            model(images, mode=mode, removed_blocks=removed_blocks)
        peak_mb = -1.0
        
    # ── 延迟测量：多次重复取均值，GPU 需要在每次前后同步 ───────────────────
    # 为什么每次推理都要 synchronize？
    # CUDA 操作是异步的：Python 调用 model(images) 后立刻返回，
    # GPU 其实还在跑。不 synchronize 的话，perf_counter 记录的是
    # "把指令提交给 GPU 的时间"，而不是"GPU 真正跑完的时间"。
    timings: list[float] = []
    with torch.no_grad():
        for _ in range(latency_reps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(images, mode=mode, removed_blocks=removed_blocks)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - t0) * 1000.0)

    mean_latency_ms = sum(timings) / len(timings)
    return peak_mb, mean_latency_ms


def main() -> None:
    args = build_parser().parse_args()
    setup = build_setup(args, compensator_name="identity")

    model  = setup["model"]
    bundle = setup["bundle"]
    device = setup["device"]
    cfg    = setup["cfg"]
    blocks = model.get_block_names()

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = Path(
        args.output
        or Path(cfg["paths"]["result_root"]) / "Exp1_Motivation" / "Motivation1_Inference_cost" / f"{current_time}_memory_cost.csv"
    )

    if device.type != "cuda":
        print("[警告] 未检测到 CUDA 设备，峰值内存列将显示 -1（不可测）。")
        print("[警告] 推理延迟对比在 CPU 上仍然有效，但效果不如 GPU 显著。\n")

    # print(f"\n[Exp1-InferenceCost] 模型：{args.model}，共 {len(blocks)} 个残差块")
    print(f"[Exp1-InferenceCost] 延迟测量：预热 {args.latency_warmup} 次，正式 {args.latency_reps} 次")
    # print(f"[Exp1-InferenceCost] 结果将写入：{output_path}\n")

    # 取第一个 batch 作为固定测量输入，整个实验使用同一批数据，
    # 这样排除了输入差异对延迟的影响，确保各组对比条件完全一致
    measure_images, _ = next(iter(bundle.val_loader))
    measure_images = measure_images.to(device)
    print(f"[Exp1-InferenceCost] 测量 batch：{measure_images.shape}（固定，保证公平）\n")

    rows: list[dict] = []

    # ── 第 0 步：full mode baseline ──────────────────────────────────────────
    print(f"[0/{len(blocks)}] 测量 baseline（full mode，保留所有残差）...")
    full_peak_mb, full_latency_ms = _measure_one(
        model, measure_images,
        mode="full", removed_blocks=None,
        device=device,
        latency_reps=args.latency_reps,
        latency_warmup=args.latency_warmup,
    )
    print(
        f"          峰值显存：{'N/A' if full_peak_mb < 0 else f'{full_peak_mb:.2f} MB'}  "
        f"延迟：{full_latency_ms:.3f} ms"
    )

    rows.append({
        "dataset":          bundle.source,
        "model":            args.model,
        "batch_size":       measure_images.size(0),
        "mode":             "full",
        "removed_count":    0,
        "removed_blocks":   "",
        # 峰值内存
        "peak_mb":          round(full_peak_mb,   2) if full_peak_mb  >= 0 else -1,
        "saved_memory_mb":  0.0,
        "saved_memory_pct": 0.0,
        # 推理延迟
        "latency_ms":       round(full_latency_ms, 4),
        "saved_latency_ms": 0.0,
        "speedup":          1.0,
    })

    # ── 第 1~N 步：逐步扩大残差删除范围 ─────────────────────────────────────
    # 策略和 run_acc_drop.py 完全一致：从最后一个块开始往前删
    # 这样三个实验的 removed_count 字段含义相同，CSV 可以直接横向对比
    for remove_count in range(1, len(blocks) + 1):
        removed = blocks[-remove_count:]
        print(
            f"[{remove_count}/{len(blocks)}] 删除最后 {remove_count} 个块"
            f"（{removed[0]} → {removed[-1]}）..."
        )

        peak_mb, latency_ms = _measure_one(
            model, measure_images,
            mode="plain", removed_blocks=removed,
            device=device,
            latency_reps=args.latency_reps,
            latency_warmup=args.latency_warmup,
        )

        # 计算节省量（若 GPU 不可用，内存相关字段填 -1）
        saved_mem_mb  = full_peak_mb - peak_mb if full_peak_mb >= 0 and peak_mb >= 0 else -1.0
        saved_mem_pct = saved_mem_mb / full_peak_mb * 100 if full_peak_mb > 0 and saved_mem_mb >= 0 else -1.0
        saved_lat_ms  = full_latency_ms - latency_ms
        # 加速比：full / plain，> 1 表示 plain 更快
        speedup       = full_latency_ms / latency_ms if latency_ms > 0 else 1.0

        mem_str = "N/A" if peak_mb < 0 else f"{peak_mb:.2f} MB (节省 {saved_mem_mb:.2f} MB)"
        print(
            f"          峰值显存：{mem_str}  "
            f"延迟：{latency_ms:.3f} ms  "
            f"加速比：{speedup:.3f}×"
        )

        rows.append({
            "dataset":          bundle.source,
            "model":            args.model,
            "batch_size":       measure_images.size(0),
            "mode":             "plain",
            "removed_count":    remove_count,
            "removed_blocks":   ",".join(removed),
            # 峰值内存
            "peak_mb":          round(peak_mb,       2) if peak_mb      >= 0 else -1,
            "saved_memory_mb":  round(saved_mem_mb,  2) if saved_mem_mb >= 0 else -1,
            "saved_memory_pct": round(saved_mem_pct, 2) if saved_mem_pct >= 0 else -1,
            # 推理延迟
            "latency_ms":       round(latency_ms,     4),
            "saved_latency_ms": round(saved_lat_ms,   4),
            "speedup":          round(speedup,         4),
        })

    saved = write_csv(output_path, rows)
    print(f"\n[Exp1-InferenceCost] 完成。结果已保存至：{saved}")
    

    # 打印一个简洁的摘要，方便快速判断实验是否得到了预期结果
    if len(rows) > 1:
        last = rows[-1]
        print(f"\n[Exp1-InferenceCost] ── 删除全部残差块时的综合收益摘要 ──")
        if last["peak_mb"] >= 0:
            print(f"  峰值显存：{rows[0]['peak_mb']:.2f} MB → {last['peak_mb']:.2f} MB"
                  f"（节省 {last['saved_memory_pct']:.1f}%）")
        print(f"  推理延迟：{rows[0]['latency_ms']:.3f} ms → {last['latency_ms']:.3f} ms"
              f"（加速比 {last['speedup']:.3f}×）")


if __name__ == "__main__":
    main()