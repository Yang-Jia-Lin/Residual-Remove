"""Src/Models_Evaluation/memory.py"""

# src/evaluation/memory.py
"""峰值内存占用测量。

GPU 和 CPU 的测量策略不同，需要分开处理：

GPU：PyTorch 内置了精确的显存追踪器（torch.cuda.max_memory_allocated），
     reset 峰值计数器后再推理，得到的就是本次 forward 的精确峰值显存。

CPU：Python 层面没有简单的"峰值内存"接口。这里用标准库的 tracemalloc
     追踪 Python 堆的分配峰值，能抓到 tensor 的 Python 对象开销，
     但不包含 PyTorch C++ 后端预先分配的内存池。
     因此 CPU 的数字是参考估算值，不如 GPU 精确。
     对于动机实验来说，我们关心的主要是相对变化（full vs plain），
     不是绝对数值，所以这个精度已经足够。
"""
from __future__ import annotations

import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator

import torch
from torch import nn


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class MemoryResult:
    """单次内存测量的结果。"""
    peak_bytes: float
    peak_mb:    float
    device:     str
    method:     str   # "cuda_allocator" | "tracemalloc"

    def __str__(self) -> str:
        return (
            f"峰值内存 [{self.device}]: "
            f"{self.peak_mb:.2f} MB  "
            f"（测量方式: {self.method}）"
        )


@dataclass
class MemoryComparison:
    """full model vs plain model 的峰值内存对比。"""
    full:      MemoryResult
    plain:     MemoryResult
    saved_mb:  float   # full 比 plain 节省的内存量（正数=plain更省，负数=没有节省）
    saved_pct: float   # 节省百分比

    def __str__(self) -> str:
        sign = "" if self.saved_mb >= 0 else "（无节省）"
        return (
            f"full  model: {self.full.peak_mb:.2f} MB\n"
            f"plain model: {self.plain.peak_mb:.2f} MB\n"
            f"节省内存   : {self.saved_mb:.2f} MB  ({self.saved_pct:.1f}%) {sign}"
        )


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

@contextmanager
def _eval_mode(model: nn.Module) -> Generator[None, None, None]:
    """临时切换到 eval 模式，退出时恢复原始状态。"""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


# ---------------------------------------------------------------------------
# 测量函数
# ---------------------------------------------------------------------------

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
    - CUDA：reset 峰值计数器后推理，结果是本次 forward 的精确峰值显存
    - CPU ：tracemalloc 估算，反映相对变化
    """
    device = sample.device

    if device.type == "cuda":
        return _measure_cuda(model, sample, device, **forward_kwargs)
    return _measure_cpu(model, sample, **forward_kwargs)


def _measure_cuda(
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    **forward_kwargs: Any,
) -> MemoryResult:
    # reset 必须在 forward 之前，否则拿到的是历史峰值而非本次峰值
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    with _eval_mode(model), torch.no_grad():
        model(sample, **forward_kwargs)

    torch.cuda.synchronize(device)
    peak_bytes = float(torch.cuda.max_memory_allocated(device))

    return MemoryResult(
        peak_bytes = peak_bytes,
        peak_mb    = peak_bytes / (1024 ** 2),
        device     = str(device),
        method     = "cuda_allocator",
    )


def _measure_cpu(
    model: nn.Module,
    sample: torch.Tensor,
    **forward_kwargs: Any,
) -> MemoryResult:
    tracemalloc.start()
    try:
        with _eval_mode(model), torch.no_grad():
            model(sample, **forward_kwargs)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    peak_bytes = float(peak_bytes)
    return MemoryResult(
        peak_bytes = peak_bytes,
        peak_mb    = peak_bytes / (1024 ** 2),
        device     = "cpu",
        method     = "tracemalloc",
    )


# ---------------------------------------------------------------------------
# 对比入口
# ---------------------------------------------------------------------------

def compare_memory(
    full_model:  nn.Module,
    plain_model: nn.Module,
    sample:      torch.Tensor,
    **forward_kwargs: Any,
) -> MemoryComparison:
    """对比 full model 和 plain model 的峰值内存占用。

    plain model 移除了残差连接，不再需要在内存中同时保留 F(x) 和 x，
    理论上峰值内存应当下降。这个函数量化这个下降幅度。

    两个 model 应当处于相同设备，sample 的 device 决定测量方式。

    Args:
        full_model:  带残差连接的原始模型。
        plain_model: 移除残差连接后的模型（可含补偿器）。
        sample:      用于 forward 的输入张量。
        **forward_kwargs: 透传给两个模型的额外参数。
    """
    full_result  = measure_peak_memory(full_model,  sample, **forward_kwargs)
    plain_result = measure_peak_memory(plain_model, sample, **forward_kwargs)

    # saved > 0 表示 plain 更省内存，saved < 0 表示 plain 反而更费内存
    saved_mb  = full_result.peak_mb - plain_result.peak_mb
    saved_pct = (saved_mb / full_result.peak_mb * 100) if full_result.peak_mb != 0 else 0.0

    return MemoryComparison(
        full      = full_result,
        plain     = plain_result,
        saved_mb  = saved_mb,
        saved_pct = saved_pct,
    )