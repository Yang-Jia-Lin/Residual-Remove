"""所有补偿器的具体实现

使用方法：
    from models.compensators import build_compensator, freeze_backbone_except_compensators

    # 构建补偿器
    comp = build_compensator("lora", channels=512, rank=16, activation="gelu")
    comp = build_compensator("affine", channels=256)
    comp = build_compensator("identity", channels=0)  # identity 不需要 channels

    # 冻结主干微调
    from models.compensators import trainable_compensator_parameters
    freeze_backbone_except_compensators(model)
    optimizer = torch.optim.Adam(trainable_compensator_parameters(model), lr=1e-3)
"""

from __future__ import annotations
import torch
from torch import nn


class BaseCompensator(nn.Module):
    """Base class for all compensator operators."""

    is_compensator = True
    fusible = False


class IdentityCompensator(BaseCompensator):
    """No-op baseline."""

    fusible = True

    def __init__(self, channels: int = 0, rank: int = 16, activation: str = "gelu") -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ScalarCompensator(BaseCompensator):
    """Single-parameter scaling baseline."""

    fusible = True

    def __init__(self, channels: int = 0, rank: int = 16, activation: str = "gelu") -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x


class AffineCompensator(BaseCompensator):
    """Per-channel affine compensator."""

    fusible = True

    def __init__(self, channels: int, rank: int = 16, activation: str = "gelu") -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x + self.beta


class Linear1x1Compensator(BaseCompensator):
    """Channel-wise 1x1 projection that can be re-parameterized into the backbone."""

    fusible = True

    def __init__(self, channels: int, rank: int = 16, activation: str = "gelu") -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.dirac_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LoRACompensator(BaseCompensator):
    """Low-rank linear compensator."""

    def __init__(self, channels: int, rank: int = 16, activation: str = "gelu") -> None:
        super().__init__()
        rank = max(1, min(rank, channels))
        self.down = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.up.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class AdapterCompensator(BaseCompensator):
    """非线性上界补偿器：y = W_2(σ(W_1(z)))

    表达力最强的补偿方案，引入了非线性变换。
    作为 Upper Bound 使用，用于衡量其他线性补偿器与理论上限的差距。

    注意：由于引入了额外的非线性计算图，这个补偿器无法像 Linear1x1 那样
    通过重参数化融入主干，部署时会有真实的计算开销（fusible=False）。
    """

    fusible = False

    def __init__(self, channels: int, rank: int = 16, activation: str = "gelu") -> None:
        super().__init__()
        rank = max(1, min(rank, channels))

        # W_1：降维投影，channels → rank
        self.down = nn.Conv2d(channels, rank, kernel_size=1, bias=True)
        # W_2：升维投影，rank → channels
        self.up   = nn.Conv2d(rank, channels, kernel_size=1, bias=True)
        self.act  = _build_activation(activation)

        # 初始化策略：
        # down 层用 kaiming 均匀初始化，bias 初始化为小值而非零，
        # 避免 dead initialization（全零输出经过激活后梯度消失）
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        if self.down.bias is not None:
            nn.init.uniform_(self.down.bias, -0.01, 0.01)

        # up 层权重用 kaiming，bias 初始化为零，
        # 让补偿器在训练初期输出接近零，不破坏预训练主干的初始分布
        nn.init.kaiming_uniform_(self.up.weight, a=5 ** 0.5)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = W_2(σ(W_1(z)))
        return self.up(self.act(self.down(x)))


LowRankCompensator = LoRACompensator


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
    """Build a compensator by name."""

    key = (name or "identity").lower()
    if key in {"identity", "none"}:
        return IdentityCompensator(channels=channels, rank=rank, activation=activation)
    if key == "scalar":
        return ScalarCompensator(channels=channels, rank=rank, activation=activation)
    if key == "affine":
        return AffineCompensator(channels=channels, rank=rank, activation=activation)
    if key in {"linear1x1", "linear_1x1", "linear"}:
        return Linear1x1Compensator(channels=channels, rank=rank, activation=activation)
    if key in {"low_rank", "lora"}:
        return LoRACompensator(channels=channels, rank=rank, activation=activation)
    # if key == "adapter":
    #     return AdapterCompensator(channels=channels, rank=rank, activation=activation)
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
