"""Src/Metrics/static_cost.py
计算模型的静态计算代价指标（参数量、MACs、参数内存占用）
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import torch
from torch import nn


@dataclass
class ModelStaticCost:
    total_params: int  # 总参数量（包含补偿器）
    backbone_params: int  # 主干参数量（不含补偿器）
    compensator_params: int  # 补偿器参数量
    macs_per_sample: int  # 单张图片的 MACs
    param_mb: float  # 参数静态内存估算（MB，float32）

    def __str__(self) -> str:
        def _fmt(n: int) -> str:
            if n >= 1_000_000_000:
                return f"{n / 1e9:.2f}G"
            if n >= 1_000_000:
                return f"{n / 1e6:.2f}M"
            if n >= 1_000:
                return f"{n / 1e3:.2f}K"
            return str(n)

        return (
            f"总参数量       : {_fmt(self.total_params)}"
            f"  (主干 {_fmt(self.backbone_params)} + 补偿器 {_fmt(self.compensator_params)})\n"
            f"MACs/样本      : {_fmt(self.macs_per_sample)}\n"
            f"参数内存 (fp32): {self.param_mb:.2f} MB"
        )


@contextmanager
def _eval_mode(model: nn.Module) -> Generator[None, None, None]:
    """临时切换到 eval 模式，退出时恢复原始状态"""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


@contextmanager
def _forward_hooks(
    model: nn.Module,
    conv_hook,
    linear_hook,
) -> Generator[None, None, None]:
    """注册 forward hook，退出时自动移除"""
    handles = [
        module.register_forward_hook(
            conv_hook if isinstance(module, nn.Conv2d) else linear_hook
        )
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# 参数量统计
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """统计模型所有参数的总数"""
    return sum(p.numel() for p in model.parameters())


def count_compensator_parameters(model: nn.Module) -> int:
    """统计补偿器参数量（通过 is_compensator 标记）"""
    seen: set[int] = set()
    total = 0
    for module in model.modules():
        if getattr(module, "is_compensator", False):
            for p in module.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    total += p.numel()
    return total


# ---------------------------------------------------------------------------
# MACs(Multiply-Accumulate Operations) 乘加运算量统计
# ---------------------------------------------------------------------------


def estimate_macs(
    model: nn.Module,
    sample: torch.Tensor,
    **forward_kwargs,
) -> int:
    """通过 forward hook 统计单张图片的 MACs"""
    batch_size = sample.size(0)
    total_macs = 0

    def conv_hook(module: nn.Conv2d, _inputs, output: torch.Tensor) -> None:
        nonlocal total_macs
        # kernel_ops = kernel_h × kernel_w × (C_in / groups)
        # groups > 1 正确处理 depthwise conv（MobileNet 等结构）
        kernel_ops = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * (module.in_channels // module.groups)
        )
        # output.shape = (N, C_out, H_out, W_out)
        total_macs += output.numel() * kernel_ops

    def linear_hook(module: nn.Linear, _inputs, output: torch.Tensor) -> None:
        nonlocal total_macs
        # output.numel() // out_features = 有效 batch token 数
        # 支持 3D 输入（batch, seq_len, dim），兼容 Transformer 结构
        total_macs += (output.numel() // module.out_features) * module.in_features

    with _eval_mode(model), _forward_hooks(model, conv_hook, linear_hook):
        with torch.no_grad():
            model(sample, **forward_kwargs)

    return int(total_macs // batch_size)


# ---------------------------------------------------------------------------
# 总入口
# ---------------------------------------------------------------------------


def analyze_model(
    model: nn.Module,
    sample: torch.Tensor,
    **forward_kwargs,
) -> ModelStaticCost:
    """一次性计算所有静态代价指标，返回结构化的 ModelStaticCost"""
    total = count_parameters(model)
    comp = count_compensator_parameters(model)
    backbone = total - comp
    macs = estimate_macs(model, sample, **forward_kwargs)
    param_mb = total * 4 / (1024**2)  # float32 = 4 bytes / param

    return ModelStaticCost(
        total_params=total,
        backbone_params=backbone,
        compensator_params=comp,
        macs_per_sample=macs,
        param_mb=param_mb,
    )
