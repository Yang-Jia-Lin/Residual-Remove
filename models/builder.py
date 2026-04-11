from __future__ import annotations

import copy

from torch import nn

from .mobilenet import ResidualMobileNetV2
from .resnet import build_resnet


def build_model(
    model_name: str,
    num_classes: int,
    input_size: int,
    compensator_name: str = "identity",
    compensator_rank: int = 16,
    adapter_activation: str = "gelu",
    width_mult: float = 1.0,
) -> nn.Module:
    key = model_name.lower()
    if key.startswith("resnet"):
        depth = int(key.replace("resnet", ""))
        return build_resnet(
            depth=depth,
            num_classes=num_classes,
            input_size=input_size,
            compensator_name=compensator_name,
            compensator_rank=compensator_rank,
            adapter_activation=adapter_activation,
        )
    if key in {"mobilenet_v2", "mobilenetv2"}:
        return ResidualMobileNetV2(
            num_classes=num_classes,
            width_mult=width_mult,
            input_size=input_size,
            compensator_name=compensator_name,
            compensator_rank=compensator_rank,
            adapter_activation=adapter_activation,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def clone_teacher_to_student(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def get_block_names(model: nn.Module) -> list[str]:
    if hasattr(model, "get_block_names"):
        return list(model.get_block_names())
    raise AttributeError(f"Model {type(model).__name__} does not expose get_block_names().")
