"""Official-style MobileNetV2 definitions and builders."""

from __future__ import annotations

from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.models.mobilenetv2 import InvertedResidual, MobileNetV2


def build_mobilenet_v2(
    num_classes: int = 1000,
    pretrained: bool = False,
    width_mult: float = 1.0,
) -> MobileNetV2:
    """Build an official torchvision MobileNetV2 and optionally load pretrained weights."""

    if pretrained and width_mult != 1.0:
        raise ValueError("Pretrained MobileNetV2 is only supported with width_mult=1.0")
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights, width_mult=width_mult)
    if num_classes != model.classifier[-1].out_features:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


__all__ = ["InvertedResidual", "MobileNetV2", "build_mobilenet_v2"]