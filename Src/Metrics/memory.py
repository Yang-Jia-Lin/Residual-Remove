"""Src/Models_Evaluation/memory.py
GPU：PyTorch 内置 torch.cuda.max_memory_allocated
CPU：Python 层面没有接口。用标准库的 tracemalloc 估算，不如 GPU 精确
"""

import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator

import torch
from torch import nn


@dataclass
class MemoryResult:
    """单次内存测量的结果。"""

    peak_bytes: float
    peak_mb: float
    device: str
    method: str  # "cuda_allocator" | "tracemalloc"

    def __str__(self) -> str:
        return (
            f"峰值内存 [{self.device}]: "
            f"{self.peak_mb:.2f} MB  "
            f"（测量方式: {self.method}）"
        )


@dataclass
class MemoryComparison:
    """full model vs plain model 的峰值内存对比。"""

    full: MemoryResult
    plain: MemoryResult
    saved_mb: float  # full 比 plain 节省的内存量（正数=plain更省，负数=没有节省）
    saved_pct: float  # 节省百分比

    def __str__(self) -> str:
        sign = "" if self.saved_mb >= 0 else "（无节省）"
        return (
            f"full  model: {self.full.peak_mb:.2f} MB\n"
            f"plain model: {self.plain.peak_mb:.2f} MB\n"
            f"节省内存   : {self.saved_mb:.2f} MB  ({self.saved_pct:.1f}%) {sign}"
        )


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


def parameter_bytes(model: nn.Module) -> int:
    """计算模型所有参数占用的字节数（仅参数，不含中间激活）。"""
    return sum(p.numel() * p.element_size() for p in model.parameters())


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
        peak_bytes=peak_bytes,
        peak_mb=peak_bytes / (1024**2),
        device=str(device),
        method="cuda_allocator",
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
        peak_bytes=peak_bytes,
        peak_mb=peak_bytes / (1024**2),
        device="cpu",
        method="tracemalloc",
    )


def measure_peak_memory(
    model: nn.Module,
    sample: torch.Tensor,
    **forward_kwargs: Any,
) -> MemoryResult:
    """测量一次前向推理的峰值内存占用"""
    device = sample.device

    if device.type == "cuda":
        return _measure_cuda(model, sample, device, **forward_kwargs)
    return _measure_cpu(model, sample, **forward_kwargs)


def compare_memory(
    full_model: nn.Module,
    plain_model: nn.Module,
    sample: torch.Tensor,
    **forward_kwargs: Any,
) -> MemoryComparison:
    """对比 full model 和 plain model 的峰值内存占用"""
    full_result = measure_peak_memory(full_model, sample, **forward_kwargs)
    plain_result = measure_peak_memory(plain_model, sample, **forward_kwargs)

    # saved > 0 表示 plain 更省内存，saved < 0 表示 plain 反而更费内存
    saved_mb = full_result.peak_mb - plain_result.peak_mb
    saved_pct = (
        (saved_mb / full_result.peak_mb * 100) if full_result.peak_mb != 0 else 0.0
    )

    return MemoryComparison(
        full=full_result,
        plain=plain_result,
        saved_mb=saved_mb,
        saved_pct=saved_pct,
    )
