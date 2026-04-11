# src/evaluation/flops.py
"""参数量与 MACs 统计（静态分析，不依赖真实数据集）。

MACs（Multiply-Accumulate Operations）是衡量模型计算量的标准指标，
这里统计的是单张图片的 MACs，而不是整个 batch 的总量。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelCost:
    """模型计算代价的结构化摘要。"""
    total_params:       int     # 总参数量（包含补偿器）
    backbone_params:    int     # 主干参数量（不含补偿器）
    compensator_params: int     # 补偿器参数量
    macs_per_sample:    int     # 单张图片的 MACs
    param_mb:           float   # 参数占用内存（MB，float32）

    def __str__(self) -> str:
        def _fmt(n: int) -> str:
            # 把大数字格式化成 K / M / G 的可读形式
            if n >= 1e9:
                return f"{n/1e9:.2f}G"
            if n >= 1e6:
                return f"{n/1e6:.2f}M"
            if n >= 1e3:
                return f"{n/1e3:.2f}K"
            return str(n)

        return (
            f"总参数量       : {_fmt(self.total_params)}"
            f"  (主干 {_fmt(self.backbone_params)} + 补偿器 {_fmt(self.compensator_params)})\n"
            f"MACs/样本      : {_fmt(self.macs_per_sample)}\n"
            f"参数内存 (fp32): {self.param_mb:.2f} MB"
        )


def count_parameters(model: nn.Module) -> int:
    """统计模型所有参数的总数量。"""
    return sum(p.numel() for p in model.parameters())


def count_compensator_parameters(model: nn.Module) -> int:
    """统计补偿器参数数量（通过 is_compensator 标记识别）。
    
    这个函数依赖 compensators.py 里约定的 is_compensator=True 标记，
    和 freeze_backbone_except_compensators 使用相同的识别机制，保持一致性。
    """
    total = 0
    for module in model.modules():
        if getattr(module, "is_compensator", False):
            total += sum(p.numel() for p in module.parameters())
    return total


def estimate_macs(
    model: nn.Module,
    sample: torch.Tensor,
    **forward_kwargs,
) -> int:
    """通过 forward hook 统计单张图片的 MACs。

    只统计 Conv2d 和 Linear 两种算子，这是 ResNet/MobileNet 的主要计算来源。
    BN、ReLU、Pooling 的计算量相对极小，忽略不计。

    注意：sample 可以是 batch，函数会自动除以 batch_size 换算为单张图片的 MACs。
    """
    hooks = []
    total_macs = 0
    # 记录 batch size，用于最后换算为 per-sample MACs
    batch_size = sample.size(0)

    def conv_hook(module: nn.Conv2d, inputs, output) -> None:
        nonlocal total_macs
        out_h, out_w = output.shape[2], output.shape[3]
        # 每个输出位置的计算量 = kernel_h × kernel_w × (in_channels / groups)
        # groups > 1 对应 depthwise conv（MobileNet 的关键结构）
        kernel_ops = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * (module.in_channels // module.groups)
        )
        # 当前 batch 的总 MACs = 输出元素数 × 每个输出位置的 kernel_ops
        # 注意这里不乘 batch_size，因为 output.shape[0] 已经包含了 batch 维度
        batch_macs = output.shape[0] * output.shape[1] * out_h * out_w * kernel_ops
        total_macs += batch_macs

    def linear_hook(module: nn.Linear, inputs, output) -> None:
        nonlocal total_macs
        # Linear 的 MACs = batch_size × in_features × out_features
        batch_macs = output.shape[0] * module.in_features * module.out_features
        total_macs += batch_macs

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(sample, **forward_kwargs)
    finally:
        # 无论推理是否成功，都要确保 hook 被移除，否则会影响后续调用
        for hook in hooks:
            hook.remove()
        if was_training:
            model.train()

    # 除以 batch_size，得到单张图片的 MACs（学术报告的标准单位）
    return int(total_macs // batch_size)


def analyze_model(
    model: nn.Module,
    sample: torch.Tensor,
    **forward_kwargs,
) -> ModelCost:
    """一次性计算所有代价指标，返回结构化的 ModelCost。"""
    total      = count_parameters(model)
    comp       = count_compensator_parameters(model)
    backbone   = total - comp
    macs       = estimate_macs(model, sample, **forward_kwargs)
    param_mb   = total * 4 / (1024 ** 2)   # float32 = 4 bytes

    return ModelCost(
        total_params       = total,
        backbone_params    = backbone,
        compensator_params = comp,
        macs_per_sample    = macs,
        param_mb           = param_mb,
    )