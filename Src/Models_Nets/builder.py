""" Src/Models_Nets/builder.py
    加载原始模型并注入补偿器（所有脚本的入口）"""
import copy
from torch import nn
from Src.Models_Nets.injector import inject, mobilenet_block_specs, resnet_block_specs
from Src.Models_Nets.origin.mobilenet import build_mobilenet_v2
from Src.Models_Nets.origin.resnet import build_resnet


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
    """构建一个官方模型，并将补偿器注入残差层"""

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
    from models.builder import build_model, get_block_names
    model1 = build_model("resnet50", num_classes=1000, pretrained=True)
    model2 = build_model("resnet50", pretrained=True, compensator_name="lora", compensator_rank=16)
    model3 = build_model("resnet50", pretrained=True, compensator_name="affine")
    model4 = build_model("mobilenet_v2", num_classes=1000, pretrained=True, compensator_name="lora")
    model5 = build_model("resnet18", num_classes=100, pretrained=False)
    names = get_block_names(model5)