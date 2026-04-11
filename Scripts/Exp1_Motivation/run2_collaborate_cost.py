"""Exp1 动机实验：量化残差连接在切分点处的完整系统传输开销。

把传输开销分解为一条完整的流水线：
  数据量 → 序列化时间 → 网络传输时间 → 总系统时间

对于每个残差块，在每种带宽场景下，同时输出 DAG 和链式拓扑的对比数字，
让审稿人一眼看出在完整端边传输流程里，删除残差分支能带来多大的系统收益。

典型运行命令：
    python experiments/Exp1_Motivation/run_system_cost.py \\
        --model resnet50 --dataset imagenet \\
        --device cuda:0 --batch-size 1 \\
        --num-workers 4 --pretrained
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from Scripts.common import add_common_args, build_setup, get_probe_batch
from Src.Collab_System.bandwidth_sim import estimate_transfer_time_ms, saved_transfer_ratio
from Src.Utils.runtime import tensor_bytes, write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="量化残差连接在完整传输流水线中的开销（Exp1 系统部分）。"
    )
    add_common_args(parser)
    parser.add_argument("--system-config", default="Configs/system.yaml")
    parser.add_argument("--output", default=None,
        help="输出 CSV 路径。不指定则写到 result_root/Exp1_Motivation/Motivation2_Collaborate_cost/system_cost.csv")
    parser.add_argument("--serialize-reps", type=int, default=20,
        help="序列化耗时的重复测量次数，取均值以减少抖动（默认 20）")
    return parser


def _measure_serialize_ms(tensor: torch.Tensor, repetitions: int) -> float:
    """实测将一个张量序列化为字节流所需的时间（毫秒），取多次均值。

    使用 torch.save 写入内存中的 BytesIO，模拟端侧设备打包特征图的真实操作。
    张量先移到 CPU，因为网络发送前必须先从 GPU 显存拷贝到内存，这个开销是真实存在的。
    """
    tensor_cpu = tensor.cpu()
    timings: list[float] = []
    for _ in range(repetitions):
        buf = io.BytesIO()
        t0 = time.perf_counter()
        torch.save(tensor_cpu, buf)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return sum(timings) / len(timings)


def main() -> None:
    args = build_parser().parse_args()
    setup = build_setup(args, compensator_name="identity")

    model  = setup["model"]
    bundle = setup["bundle"]
    device = setup["device"]
    cfg    = setup["cfg"]

    sys_cfg_path = Path(args.system_config)
    if not sys_cfg_path.exists():
        raise FileNotFoundError(f"找不到系统配置文件：{sys_cfg_path}")
    sys_cfg = yaml.safe_load(sys_cfg_path.read_text(encoding="utf-8"))

    bandwidth_list       = [float(b) for b in sys_cfg["bandwidth_mbps"]]
    protocol_overhead_ms = float(sys_cfg["protocol_overhead_ms"])

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = Path(
        args.output
        or Path(cfg["paths"]["result_root"]) / "Exp1_Motivation" / "Motivation2_Collaborate_cost" / f"{current_time}_system_cost.csv"
    )

    print(f"\n[Exp1-System] 模型：{args.model}")
    print(f"[Exp1-System] 带宽场景：{bandwidth_list} Mbps")
    print(f"[Exp1-System] 协议固定开销：{protocol_overhead_ms} ms")
    print(f"[Exp1-System] 序列化重复次数：{args.serialize_reps}")
    print(f"[Exp1-System] 结果将写入：{output_path}\n")

    # batch_size=1：关心单张图片的张量大小，不需要 batch 维度的统计意义
    images, _ = get_probe_batch(bundle, device, batch_size=1)

    print("[Exp1-System] 运行 full mode 前向，收集中间张量...")
    model.eval()
    with torch.no_grad():
        output = model(images, mode="full", return_residual_stats=True)

    residual_stats = output["residual_stats"]
    blocks = list(residual_stats.keys())
    print(f"[Exp1-System] 采集到 {len(blocks)} 个残差块\n")

    # ── 第一步：对每个 block，先把序列化耗时测好（和带宽无关，只测一次）──────
    # 这一步放在外层循环外面，避免每种带宽场景都重复测序列化时间
    print("[Exp1-System] 正在测量序列化耗时（每个块需要约数秒）...")
    serialize_cache: dict[str, tuple[float, float]] = {}

    for block_name, stats in residual_stats.items():
        # DAG：需要分别序列化 F(x) 和 x 两路
        dag_plain_ms    = _measure_serialize_ms(stats["plain"],    args.serialize_reps)
        dag_identity_ms = _measure_serialize_ms(stats["identity"], args.serialize_reps)
        dag_serialize_ms = dag_plain_ms + dag_identity_ms

        # 链式：只序列化 F(x) 一路
        chain_serialize_ms = _measure_serialize_ms(stats["plain"], args.serialize_reps)

        serialize_cache[block_name] = (dag_serialize_ms, chain_serialize_ms)
        print(
            f"  {block_name}: "
            f"DAG 序列化={dag_serialize_ms:.3f} ms  "
            f"链式序列化={chain_serialize_ms:.3f} ms  "
            f"节省={dag_serialize_ms - chain_serialize_ms:.3f} ms"
        )

    # ── 第二步：组合所有维度，每行是 (block × bandwidth) ───────────────────────
    print(f"\n[Exp1-System] 组合数据量 + 序列化 + 传输延迟 + 总系统时间...\n")

    rows: list[dict] = []

    for block_name, stats in residual_stats.items():
        # ── 维度 1：数据量（和带宽无关，每个 block 固定）───────────────────
        dag_bytes   = tensor_bytes(stats["plain"]) + tensor_bytes(stats["identity"])
        chain_bytes = tensor_bytes(stats["plain"])
        saved_bytes = dag_bytes - chain_bytes
        size_ratio  = saved_transfer_ratio(dag_bytes, chain_bytes)

        # ── 维度 2：序列化时间（已经测好，从缓存读取）────────────────────────
        dag_serialize_ms, chain_serialize_ms = serialize_cache[block_name]
        saved_serialize_ms = dag_serialize_ms - chain_serialize_ms

        for bandwidth_mbps in bandwidth_list:
            # ── 维度 3：理论网络传输时间（取决于带宽）──────────────────────
            dag_transfer_ms   = estimate_transfer_time_ms(dag_bytes,   bandwidth_mbps, protocol_overhead_ms)
            chain_transfer_ms = estimate_transfer_time_ms(chain_bytes, bandwidth_mbps, protocol_overhead_ms)
            saved_transfer_ms = dag_transfer_ms - chain_transfer_ms

            # ── 维度 4：总系统时间 = 序列化 + 网络传输 ──────────────────────
            # 这是端侧视角的完整等待时间：先打包数据，再等网络传完
            dag_total_ms   = dag_serialize_ms   + dag_transfer_ms
            chain_total_ms = chain_serialize_ms + chain_transfer_ms
            saved_total_ms = dag_total_ms - chain_total_ms
            total_saved_pct = saved_total_ms / dag_total_ms * 100 if dag_total_ms > 0 else 0.0

            rows.append({
                "dataset":              bundle.source,
                "model":                args.model,
                "block":                block_name,
                "bandwidth_mbps":       bandwidth_mbps,

                # 维度 1：数据量对比
                "dag_bytes":            dag_bytes,
                "chain_bytes":          chain_bytes,
                "saved_bytes":          saved_bytes,
                "size_saved_pct":       round(size_ratio * 100, 2),

                # 维度 2：序列化时间对比
                "dag_serialize_ms":     round(dag_serialize_ms,   4),
                "chain_serialize_ms":   round(chain_serialize_ms, 4),
                "saved_serialize_ms":   round(saved_serialize_ms, 4),

                # 维度 3：理论网络传输时间对比
                "dag_transfer_ms":      round(dag_transfer_ms,   4),
                "chain_transfer_ms":    round(chain_transfer_ms, 4),
                "saved_transfer_ms":    round(saved_transfer_ms, 4),

                # 维度 4：总系统时间对比（核心对比指标）
                "dag_total_ms":         round(dag_total_ms,    4),
                "chain_total_ms":       round(chain_total_ms,  4),
                "saved_total_ms":       round(saved_total_ms,  4),
                "total_saved_pct":      round(total_saved_pct, 2),
            })

    saved = write_csv(output_path, rows)
    print(f"[Exp1-System] 完成。结果已保存至：{saved}")
    print(
        f"[Exp1-System] 共 {len(rows)} 行"
        f" = {len(blocks)} 块 × {len(bandwidth_list)} 种带宽场景"
    )


if __name__ == "__main__":
    main()