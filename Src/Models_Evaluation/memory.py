# src/evaluation/memory.py
"""峰值内存占用测量。

GPU 和 CPU 的测量策略完全不同，必须分开处理：

GPU：PyTorch 内置了精确的显存追踪器（torch.cuda.max_memory_allocated），
     可以直接得到精确的峰值显存数字。

CPU：Python 层面没有简单的"峰值内存"接口。这里用标准库的 tracemalloc
     追踪 Python 堆的分配峰值，这能抓到 Tensor 的 Python 对象开销，
     但不包含 PyTorch C++ 后端预先分配的内存池。
     因此 CPU 的数字是一个参考估算值，不如 GPU 精确。
     对于动机实验来说，我们关心的主要是相对变化（full vs plain），
     不是绝对数值，所以这个精度已经足够。
"""
from __future__ import annotations

import tracemalloc
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class MemoryResult:
    """单次内存测量的结果。"""
    peak_bytes: float
    peak_mb:    float
    device:     str
    method:     str   # "cuda_allocator" | "tracemalloc"（说明测量方式）

    def __str__(self) -> str:
        return (
            f"峰值内存 [{self.device}]: "
            f"{self.peak_mb:.2f} MB  "
            f"（测量方式: {self.method}）"
        )


@dataclass
class MemoryComparison:
    """full mode vs plain mode 的峰值内存对比。"""
    full:      MemoryResult
    plain:     MemoryResult
    saved_mb:  float   # plain 比 full 节省的内存量（可为负数，即没有节省）
    saved_pct: float   # 节省百分比

    def __str__(self) -> str:
        return (
            f"full  mode: {self.full.peak_mb:.2f} MB\n"
            f"plain mode: {self.plain.peak_mb:.2f} MB\n"
            f"节省内存   : {self.saved_mb:.2f} MB  ({self.saved_pct:.1f}%)"
        )


def parameter_bytes(model: nn.Module) -> int:
    """计算模型所有参数占用的字节数（仅参数，不含中间激活）。"""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def measure_peak_memory(
    model: nn.Module,
    sample: torch.Tensor,
    **forward_kwargs: Any,
) -> MemoryResult:
    """测量一次前向推理的峰值内存占用。

    根据设备类型自动选择测量方式：
    - CUDA：使用 torch.cuda.max_memory_allocated（精确）
    - CPU： 使用 tracemalloc（估算，反映相对变化）
    """
    model.eval()
    device = sample.device

    if device.type == "cuda":
        # ── GPU 路径：PyTorch 原生显存追踪 ───────────────────────────────
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model(sample, **forward_kwargs)
        peak_bytes = float(torch.cuda.max_memory_allocated(device))
        return MemoryResult(
            peak_bytes = peak_bytes,
            peak_mb    = peak_bytes / (1024 ** 2),
            device     = str(device),
            method     = "cuda_allocator",
        )

    # ── CPU 路径：tracemalloc 追踪 Python 堆分配 ─────────────────────────
    # 注意：tracemalloc 追踪的是 Python 层面的分配，
    # 不包含 PyTorch 内存池中已经预分配的部分。
    # 但对于 full vs plain 的相对对比，这个精度是足够的。
    tracemalloc.start()
    try:
        with torch.no_grad():
            model(sample, **forward_kwargs)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    peak_bytes = float(peak_bytes)
    return MemoryResult(
        peak_bytes = peak_bytes,
        peak_mb    = peak_bytes / (1024 ** 2),
        device     = str(device),
        method     = "tracemalloc",
    )


def compare_memory(
    model: nn.Module,
    sample: torch.Tensor,
    removed_blocks: list[str] | None = None,
) -> MemoryComparison:
    """对比 full 和 plain mode 的峰值内存占用。

    plain mode 删除了残差连接，不再需要在内存中同时保留 F(x) 和 x，
    理论上峰值内存应当下降。这个函数量化这个下降幅度。
    """
    full_result  = measure_peak_memory(model, sample, mode="full")
    plain_result = measure_peak_memory(
        model, sample, mode="plain", removed_blocks=removed_blocks
    )

    saved_mb = plain_result.peak_mb - full_result.peak_mb
    saved_pct = (saved_mb / full_result.peak_mb * 100) if full_result.peak_mb != 0 else 0.0

    return MemoryComparison(
        full=full_result,
        plain=plain_result,
        saved_mb=saved_mb,
        saved_pct=saved_pct,
    )