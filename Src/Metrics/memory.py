"""Src/Metrics/memory.py
GPU：PyTorch 内置 torch.cuda.max_memory_allocated
CPU：Python 层面没有接口。用标准库的 tracemalloc 估算，不如 GPU 精确
现在都不准确，后续再完善
"""

import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator

import torch
from torch import nn

# ══════════════════════════════════════════════════════════════════════════════
# § 1  普通显存峰值占用
# ══════════════════════════════════════════════════════════════════════════════


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
    torch.cuda.empty_cache()
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


# ══════════════════════════════════════════════════════════════════════════════
# § 2  最大可用 batch size
# ══════════════════════════════════════════════════════════════════════════════


def find_max_batch_size(
    model: nn.Module,
    sample_single: torch.Tensor,  # shape [1, C, H, W]，单张图
    min_bs: int = 1,
    max_bs: int = 1024,
    **forward_kwargs: Any,
) -> int:
    """
    二分查找：在当前显存下该配置最大能跑多大的 batch
    返回最大可用 batch size，OOM 则返回 0
    """
    model.eval()

    def can_run(bs: int) -> bool:
        torch.cuda.empty_cache()
        batch = sample_single.repeat(bs, 1, 1, 1)  # 真实分配 bs 份内存
        try:
            with torch.no_grad():
                model(batch, **forward_kwargs)
            return True
        except torch.cuda.OutOfMemoryError:
            return False

    # 先确认 min_bs 可行
    if not can_run(min_bs):
        return 0

    # 二分查找
    lo, hi = min_bs, max_bs
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if can_run(mid):
            lo = mid
        else:
            hi = mid - 1

    return lo


# ══════════════════════════════════════════════════════════════════════════════
# § 3  激活内存峰值
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ActivationMemoryResult:
    """纯激活内存峰值（已扣除模型权重占用）"""

    activation_peak_mb: float
    weight_mb: float
    total_peak_mb: float
    device: str

    def __str__(self) -> str:
        return (
            f"激活内存峰值 [{self.device}]: "
            f"{self.activation_peak_mb:.2f} MB  "
            f"（权重基线: {self.weight_mb:.2f} MB，总峰值: {self.total_peak_mb:.2f} MB）"
        )


def measure_activation_memory(
    model: nn.Module,
    sample: torch.Tensor,
    **forward_kwargs: Any,
) -> ActivationMemoryResult:
    """
    测量一次前向推理的纯激活内存峰值。
    激活内存 = 总峰值 - 权重基线（forward 前已分配的显存）
    只支持 CUDA，CPU 返回 -1。
    """
    device = sample.device
    if device.type != "cuda":
        return ActivationMemoryResult(
            activation_peak_mb=-1.0,
            weight_mb=-1.0,
            total_peak_mb=-1.0,
            device="cpu",
        )

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    # forward 前：只有权重，没有激活
    weight_mb = torch.cuda.memory_allocated(device) / (1024**2)

    torch.cuda.reset_peak_memory_stats(device)
    with _eval_mode(model), torch.no_grad():
        model(sample, **forward_kwargs)
    torch.cuda.synchronize(device)

    total_peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    activation_peak_mb = total_peak_mb - weight_mb

    return ActivationMemoryResult(
        activation_peak_mb=activation_peak_mb,
        weight_mb=weight_mb,
        total_peak_mb=total_peak_mb,
        device=str(device),
    )
