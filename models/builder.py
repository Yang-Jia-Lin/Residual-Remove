"""加载原始模型并注入补偿器（所有脚本的入口）
    from models.builder import build_model, get_block_names
    model1 = build_model("resnet50", num_classes=1000, pretrained=True)
    model2 = build_model("resnet50", pretrained=True, compensator_name="lora", compensator_rank=16)
    model3 = build_model("resnet50", pretrained=True, compensator_name="affine")
    model4 = build_model("mobilenet_v2", num_classes=1000, pretrained=True, compensator_name="lora")
    model5 = build_model("resnet18", num_classes=100, pretrained=False)
    names = get_block_names(model5)
"""

from __future__ import annotations
import copy
from typing import Any
from torch import nn
from .injector import inject, mobilenet_block_specs, resnet_block_specs
from .origin.mobilenet import build_mobilenet_v2
from .origin.resnet import build_resnet


def _normalize_arch(arch: str | None, model_name: str | None) -> str:
    key = (arch or model_name or "").lower()
    if key in {"resnet18", "resnet34", "resnet50"}:
        return key
    if key in {"resnet", "resnet18"}:
        return "resnet18"
    if key in {"mobilenet", "mobilenet_v2", "mobilenetv2"}:
        return "mobilenet_v2"
    if key:
        return key
    raise ValueError("An architecture name is required.")


def build_model(
    model_name: str | None = None,
    num_classes: int = 1000,
    compensator_name: str = "identity",
    compensator_rank: int = 16,
    adapter_activation: str = "gelu",
    width_mult: float = 1.0,
    *,
    arch: str | None = None,
    depth: int | None = None,
    pretrained: bool = False,
    rank: int | None = None,
    activation: str | None = None,
) -> nn.Module:
    """Build an official model and inject compensators into residual blocks.

    The signature keeps backward compatibility with the current experiment scripts while
    also supporting the new arch/depth/pretrained-oriented entry point.
    """

    resolved_arch = _normalize_arch(arch, model_name)
    resolved_rank = int(rank if rank is not None else compensator_rank)
    resolved_activation = activation or adapter_activation

    if resolved_arch.startswith("resnet"):
        resolved_depth = depth or int(resolved_arch.replace("resnet", ""))
        backbone = build_resnet(depth=resolved_depth, num_classes=num_classes, pretrained=pretrained)
        return inject(
            backbone,
            block_specs=resnet_block_specs,
            compensator_name=compensator_name,
            rank=resolved_rank,
            activation=resolved_activation,
        )

    if resolved_arch in {"mobilenet_v2", "mobilenetv2"}:
        backbone = build_mobilenet_v2(
            num_classes=num_classes,
            pretrained=pretrained,
            width_mult=width_mult,
        )
        return inject(
            backbone,
            block_specs=mobilenet_block_specs,
            compensator_name=compensator_name,
            rank=resolved_rank,
            activation=resolved_activation,
        )

    raise ValueError(f"Unsupported model: {resolved_arch}")


def clone_teacher_to_student(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def get_block_names(model: nn.Module) -> list[str]:
    if hasattr(model, "get_block_names"):
        return list(model.get_block_names())
    raise AttributeError(f"Model {type(model).__name__} does not expose get_block_names().")



if __name__ == "__main__":
    import torch

    print("=== 开始测试 builder.py ===")
    
    # 1. 实例化模型（使用 resnet18 加快测试速度，不下载预训练权重）
    model_name = "resnet18"
    print(f"1. 正在构建 {model_name} 并注入动态路由...")
    model = build_model(model_name, num_classes=1000, pretrained=False)
    model.eval()  # 设置为推理模式

    # 2. 获取并打印切分点/残差块名称，验证注入是否成功
    blocks = get_block_names(model)
    print(f"2. 成功获取到 {len(blocks)} 个残差块。")
    print(f"   前三个 Block: {blocks[:3]}")
    print(f"   后三个 Block: {blocks[-3:]}")

    # 3. 构造虚拟输入数据 (BatchSize=1, Channels=3, Height=224, Width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"3. 构造虚拟输入图像，形状为: {dummy_input.shape}")

    # 4. 执行前向推理测试
    print("4. 开始前向推理...")
    with torch.no_grad():
        # 测试 1：作为普通 ResNet 进行全量推理（Baseline）
        out_full = model(dummy_input, mode="full")
        print(f"   [普通模式] 输出形状: {out_full.shape} (期望为 [1, 1000])")
        
        # 测试 2：测试你的核心动机实验特性 —— 强行删掉最后一个 Block 的残差
        target_remove = [blocks[-1]]
        out_plain = model(dummy_input, mode="plain", removed_blocks=target_remove)
        print(f"   [删残差模式] 移除了 {target_remove}，输出形状: {out_plain.shape}")

    print("=== 测试顺利完成！===")