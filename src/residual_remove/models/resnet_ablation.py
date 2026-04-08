from __future__ import annotations

import random
from dataclasses import dataclass
from types import MethodType
from typing import Dict, List, Tuple

import torch
from torch import nn
from torchvision.models import resnet18, resnet50
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


@dataclass
class ResidualBlockInfo:
    name: str
    stage: int
    keep_residual: bool
    lifetime_ops: int
    residual_elements: int = 0


def _basicblock_forward(self: BasicBlock, x: torch.Tensor) -> torch.Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    if getattr(self, "keep_residual", True):
        out = out + identity

    out = self.relu(out)
    return out


def _bottleneck_forward(self: Bottleneck, x: torch.Tensor) -> torch.Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    if getattr(self, "keep_residual", True):
        out = out + identity

    out = self.relu(out)
    return out


def _iter_residual_blocks(model: ResNet) -> List[Tuple[str, int, nn.Module]]:
    blocks: List[Tuple[str, int, nn.Module]] = []
    for stage, layer_name in enumerate(["layer1", "layer2", "layer3", "layer4"], start=1):
        layer = getattr(model, layer_name)
        for idx, block in enumerate(layer):
            blocks.append((f"{layer_name}.{idx}", stage, block))
    return blocks


def _estimate_lifetime_ops(block: nn.Module) -> int:
    if isinstance(block, BasicBlock):
        return 6 if block.downsample is None else 8
    if isinstance(block, Bottleneck):
        return 9 if block.downsample is None else 11
    return 0


def _patch_block_forward(block: nn.Module) -> None:
    if isinstance(block, BasicBlock):
        block.forward = MethodType(_basicblock_forward, block)
    elif isinstance(block, Bottleneck):
        block.forward = MethodType(_bottleneck_forward, block)
    else:
        raise TypeError(f"Unsupported residual block type: {type(block)!r}")


def _choose_blocks_to_remove(
    blocks: List[Tuple[str, int, nn.Module]],
    mode: str,
    value: float,
    seed: int,
) -> Dict[str, bool]:
    keep_map = {name: True for name, _, _ in blocks}
    if mode == "none":
        return keep_map
    if mode == "full":
        return {name: False for name, _, _ in blocks}
    if mode == "random_ratio":
        generator = random.Random(seed)
        names = [name for name, _, _ in blocks]
        remove_count = min(len(names), max(0, round(len(names) * value)))
        for name in generator.sample(names, k=remove_count):
            keep_map[name] = False
        return keep_map
    if mode == "stage_progressive":
        max_stage = int(value)
        return {name: stage > max_stage for name, stage, _ in blocks}
    raise ValueError(f"Unsupported ablation mode: {mode}")


def _build_backbone(model_name: str, num_classes: int) -> ResNet:
    if model_name == "resnet18":
        return resnet18(weights=None, num_classes=num_classes)
    if model_name == "resnet50":
        return resnet50(weights=None, num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def build_resnet_with_ablation(
    model_name: str,
    num_classes: int,
    ablation_mode: str = "none",
    ablation_value: float = 0.0,
    seed: int = 42,
) -> Tuple[ResNet, List[ResidualBlockInfo]]:
    model = _build_backbone(model_name=model_name, num_classes=num_classes)
    blocks = _iter_residual_blocks(model)
    keep_map = _choose_blocks_to_remove(blocks, ablation_mode, ablation_value, seed)
    block_infos: List[ResidualBlockInfo] = []

    for name, stage, block in blocks:
        _patch_block_forward(block)
        keep_residual = keep_map[name]
        block.keep_residual = keep_residual
        block.block_name = name
        block.block_stage = stage
        block.lifetime_ops = _estimate_lifetime_ops(block)
        block_infos.append(
            ResidualBlockInfo(
                name=name,
                stage=stage,
                keep_residual=keep_residual,
                lifetime_ops=block.lifetime_ops,
            )
        )

    return model, block_infos
