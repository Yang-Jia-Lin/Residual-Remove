"""Official-style ResNet definitions and builders."""

from __future__ import annotations

from typing import Any

from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


_RESNET_BUILDERS: dict[int, tuple[Any, Any]] = {
    18: (resnet18, ResNet18_Weights),
    34: (resnet34, ResNet34_Weights),
    50: (resnet50, ResNet50_Weights),
}


def build_resnet(depth: int, num_classes: int = 1000, pretrained: bool = False) -> ResNet:
    """Build an official torchvision ResNet and optionally load pretrained weights."""

    if depth not in _RESNET_BUILDERS:
        raise ValueError(f"Unsupported ResNet depth: {depth}")
    builder, weights_enum = _RESNET_BUILDERS[depth]
    weights = weights_enum.DEFAULT if pretrained else None
    if pretrained:
        model = builder(weights=weights)
        if num_classes != model.fc.out_features:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    return builder(weights=None, num_classes=num_classes)


__all__ = ["BasicBlock", "Bottleneck", "ResNet", "build_resnet"]