from __future__ import annotations

import torch
from torch import nn


class BaseCompensator(nn.Module):
    is_compensator = True
    fusible = False


class IdentityCompensator(BaseCompensator):
    fusible = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ScalarCompensator(BaseCompensator):
    fusible = True

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x


class AffineCompensator(BaseCompensator):
    fusible = True

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x + self.beta


class Linear1x1Compensator(BaseCompensator):
    fusible = True

    def __init__(self, channels: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        nn.init.dirac_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LowRankCompensator(BaseCompensator):
    def __init__(self, channels: int, rank: int = 16) -> None:
        super().__init__()
        rank = max(1, min(rank, channels))
        self.down = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.up.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class AdapterCompensator(BaseCompensator):
    def __init__(self, channels: int, rank: int = 16, activation: str = "gelu") -> None:
        super().__init__()
        rank = max(1, min(rank, channels))
        self.down = nn.Conv2d(channels, rank, kernel_size=1, bias=True)
        self.up = nn.Conv2d(rank, channels, kernel_size=1, bias=True)
        self.act = _build_activation(activation)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.down.bias)
        nn.init.kaiming_uniform_(self.up.weight, a=5**0.5)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


def _build_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU(inplace=True)
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported adapter activation: {name}")


def build_compensator(
    name: str | None,
    channels: int,
    rank: int = 16,
    activation: str = "gelu",
) -> BaseCompensator:
    key = (name or "identity").lower()
    if key in {"identity", "none"}:
        return IdentityCompensator()
    if key == "scalar":
        return ScalarCompensator()
    if key == "affine":
        return AffineCompensator(channels)
    if key in {"linear1x1", "linear_1x1", "linear"}:
        return Linear1x1Compensator(channels)
    if key in {"low_rank", "lora"}:
        return LowRankCompensator(channels, rank=rank)
    if key == "adapter":
        return AdapterCompensator(channels, rank=rank, activation=activation)
    raise ValueError(f"Unsupported compensator: {name}")


def freeze_backbone_except_compensators(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if getattr(module, "is_compensator", False):
            for parameter in module.parameters():
                parameter.requires_grad = True


def trainable_compensator_parameters(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in model.modules():
        if getattr(module, "is_compensator", False):
            params.extend(parameter for parameter in module.parameters() if parameter.requires_grad)
    return params
