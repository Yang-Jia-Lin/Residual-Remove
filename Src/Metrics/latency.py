"""Src/Models_Evaluation/latency.py
1. 预热（warmup）：GPU 的 JIT 编译和 CUDA kernel 初始化会在前几次推理中发生
2. CUDA 同步：CUDA 的操作是异步的，须用 torch.cuda.synchronize() 等待 GPU 真正跑完
3. 多次重复取均值：单次测量噪声很大，重复 20~50 次取均值稳定
"""

import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LatencyResult:
    """单次延迟结果"""

    mean_ms: float  # 平均延迟（毫秒）
    min_ms: float  # 最小延迟（毫秒，硬件上限）
    max_ms: float  # 最大延迟（毫秒）
    std_ms: float  # 标准差
    device: str

    def __str__(self) -> str:
        return (
            f"延迟 [{self.device}]: "
            f"均值 {self.mean_ms:.3f} ms  "
            f"最小 {self.min_ms:.3f} ms  "
            f"std ±{self.std_ms:.3f} ms"
        )


@dataclass
class LatencyComparison:
    """full mode vs plain mode 延迟对比"""

    full: LatencyResult
    plain: LatencyResult
    speedup: float  # plain / full 的加速比，> 1 说明 plain 更快

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
    """测量单种推理配置下的延迟，返回均值/最小值/标准差"""
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
        mean_ms=sum(timings) / len(timings),
        min_ms=min(timings),
        max_ms=max(timings),
        std_ms=statistics.stdev(timings),
        device=str(device),
    )


def compare_latency(
    full_model: torch.nn.Module,
    plain_model: torch.nn.Module,
    sample: torch.Tensor,
    repetitions: int = 50,
    warmup: int = 10,
    **forward_kwargs: Any,
) -> LatencyComparison:
    full_result = measure_latency(
        full_model, sample, repetitions, warmup, **forward_kwargs
    )
    plain_result = measure_latency(
        plain_model, sample, repetitions, warmup, **forward_kwargs
    )
    speedup = (
        full_result.mean_ms / plain_result.mean_ms if plain_result.mean_ms > 0 else 1.0
    )
    return LatencyComparison(full=full_result, plain=plain_result, speedup=speedup)
