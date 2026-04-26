"""Src/Models_Nets/compensators.py"""

import torch
from torch import nn


class BaseCompensator(nn.Module):
    is_compensator = True


class IdentityCompensator(BaseCompensator):
    def __init__(
        self, channels: int = 0, rank: int = 16, activation: str = "gelu"
    ) -> None:
        super().__init__()
        del channels, rank, activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class AffineCompensator(BaseCompensator):
    def __init__(self, channels: int, rank: int = 16, activation: str = "gelu") -> None:
        super().__init__()
        del rank, activation
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x + self.beta


class Linear1x1Compensator(BaseCompensator):
    def __init__(self, channels: int, rank: int = 16, activation: str = "gelu") -> None:
        super().__init__()
        del rank, activation
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(x)


class LoRACompensator(BaseCompensator):
    def __init__(self, channels: int, rank: int = 16, activation: str = "gelu") -> None:
        super().__init__()
        del activation
        rank = _clamp_rank(rank, channels)
        self.down = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, channels, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.down(x))


class AdapterCompensator(BaseCompensator):
    def __init__(self, channels: int, rank: int = 16, activation: str = "gelu") -> None:
        super().__init__()
        rank = _clamp_rank(rank, channels)

        self.down = nn.Conv2d(channels, rank, kernel_size=1, bias=True)
        self.up = nn.Conv2d(rank, channels, kernel_size=1, bias=True)
        self.act = _build_activation(activation)

        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        if self.down.bias is not None:
            nn.init.uniform_(self.down.bias, -0.01, 0.01)

        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


def _clamp_rank(rank: int, channels: int) -> int:
    return max(1, min(rank, channels))


def _build_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU(inplace=True)
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def build_compensator(
    name: str | None,
    channels: int,
    rank: int = 16,
    activation: str = "gelu",
) -> BaseCompensator:
    key = (name or "identity").lower()

    if key in {"identity", "none"}:
        return IdentityCompensator(channels=channels, rank=rank, activation=activation)
    if key == "affine":
        return AffineCompensator(channels=channels, rank=rank, activation=activation)
    if key in {"linear1x1", "linear_1x1", "linear"}:
        return Linear1x1Compensator(channels=channels, rank=rank, activation=activation)
    if key in {"low_rank", "lora"}:
        return LoRACompensator(channels=channels, rank=rank, activation=activation)
    if key == "adapter":
        return AdapterCompensator(channels=channels, rank=rank, activation=activation)

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
            params.extend(p for p in module.parameters() if p.requires_grad)
    return params
