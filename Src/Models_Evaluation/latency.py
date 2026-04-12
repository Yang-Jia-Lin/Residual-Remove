# src/evaluation/latency.py
"""推理延迟测量工具。

GPU 延迟测量的三个关键点：
  1. 预热（warmup）：GPU 的 JIT 编译和 CUDA kernel 初始化会在前几次推理中发生，
                     如果不预热就开始计时，数据会严重偏高。
  2. CUDA 同步：CUDA 的操作是异步的，time.perf_counter() 记录的是"提交命令"的时间，
                而不是"执行完毕"的时间。必须用 torch.cuda.synchronize() 等待 GPU 真正跑完。
  3. 多次重复取均值：单次测量噪声很大，重复 20~50 次取均值才稳定。
"""
from __future__ import annotations

import time
import statistics
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LatencyResult:
    """单次延迟测量的结果。"""
    mean_ms:   float   # 平均延迟（毫秒）
    min_ms:    float   # 最小延迟（毫秒，代表硬件上限）
    max_ms:    float   # 最大延迟（毫秒）
    std_ms:    float   # 标准差（衡量测量稳定性）
    device:    str

    def __str__(self) -> str:
        return (
            f"延迟 [{self.device}]: "
            f"均值 {self.mean_ms:.3f} ms  "
            f"最小 {self.min_ms:.3f} ms  "
            f"std ±{self.std_ms:.3f} ms"
        )


@dataclass
class LatencyComparison:
    """full mode vs plain mode 延迟对比，动机实验的核心输出。"""
    full:        LatencyResult
    plain:       LatencyResult
    speedup:     float   # plain / full 的加速比，> 1 说明 plain 更快

    def __str__(self) -> str:
        return (
            f"full  mode: {self.full.mean_ms:.3f} ms\n"
            f"plain mode: {self.plain.mean_ms:.3f} ms\n"
            f"加速比     : {self.speedup:.3f}×  "
            f"({'plain 更快 ✓' if self.speedup > 1 else 'full 更快'})"
        )


def measure_latency(
    model: torch.nn.Module,
    sample: torch.Tensor,
    repetitions: int = 50,
    warmup: int = 10,
    **forward_kwargs: Any,
) -> LatencyResult:
    """测量单种推理配置下的延迟，返回均值/最小值/标准差。

    Args:
        repetitions: 正式测量的重复次数，越大结果越稳定，建议 ≥ 20。
        warmup:      预热次数，CPU 上建议 5，GPU 上建议 10。
    """
    model.eval()
    device = sample.device

    with torch.no_grad():
        # ── 预热阶段 ─────────────────────────────────────────────────────
        for _ in range(warmup):
            model(sample, **forward_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # ── 正式计时阶段 ──────────────────────────────────────────────────
        timings: list[float] = []
        for _ in range(repetitions):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(sample, **forward_kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - t0) * 1000.0)

    return LatencyResult(
        mean_ms = sum(timings) / len(timings),
        min_ms  = min(timings),
        max_ms  = max(timings),
        std_ms  = statistics.stdev(timings),
        device  = str(device),
    )


def compare_latency(
    full_model:  torch.nn.Module,
    plain_model: torch.nn.Module,
    sample:      torch.Tensor,
    repetitions: int = 50,
    warmup:      int = 10,
    **forward_kwargs: Any,
) -> LatencyComparison:
    full_result  = measure_latency(full_model,  sample, repetitions, warmup, **forward_kwargs)
    plain_result = measure_latency(plain_model, sample, repetitions, warmup, **forward_kwargs)
    speedup = full_result.mean_ms / plain_result.mean_ms if plain_result.mean_ms > 0 else 1.0
    return LatencyComparison(full=full_result, plain=plain_result, speedup=speedup)