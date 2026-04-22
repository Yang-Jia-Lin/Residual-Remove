"""Src/Models_Nets/compensators.py"""

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

初始化原则（所有带权重的补偿器都遵守）：
    训练起点必须严格等价于 plain 模型，即补偿器初始输出 = 输入 x。
    实现方式：最后一层权重 zero-init + forward 里加残差 x +。
    这样 delta = 0，模型从 plain 的 55% 出发，后续优化只是在残差上做加法，不会崩。
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
    """Single-parameter scaling baseline.

    初始 alpha=1，forward = 1 * x = x，天然等价于 identity，无需额外处理。
    表达力极弱（全局一个标量），预期恢复效果有限，作为下界参考。
    """

    fusible = True

    def __init__(self, channels: int = 0, rank: int = 16, activation: str = "gelu") -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x


class AffineCompensator(BaseCompensator):
    """Per-channel affine compensator.

    初始 gamma=1, beta=0，forward = 1*x + 0 = x，天然等价于 identity，无需额外处理。
    表达力略强于 scalar，可与前置 BN 融合（fusible=True），部署零开销。
    """

    fusible = True

    def __init__(self, channels: int, rank: int = 16, activation: str = "gelu") -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x + self.beta


class Linear1x1Compensator(BaseCompensator):
    """Channel-wise 1x1 projection + 残差连接。

    forward = x + proj(x)，初始 proj 权重全零，即 proj(x)=0，起点严格等于 plain。

    原来用 dirac_ 初始化（让 proj(x) ≈ x，forward ≈ 2x）是错的：
      - dirac_ 不是精确 identity（Conv2d 有 bias，且 dirac_ 对非方阵不精确）
      - 初始输出 ≈ 2x 会放大特征分布，导致 loss 极大，梯度爆炸
    改成 zero-init + 残差后，初始输出 = x，训练稳定。

    融合方式：将 proj 权重加到主干最后一个 Conv 上，实现零部署开销（fusible=True）。
    """

    fusible = True

    def __init__(self, channels: int, rank: int = 16, activation: str = "gelu") -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        # zero-init：初始 delta = proj(x) = 0，forward = x + 0 = x
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(x)


class LoRACompensator(BaseCompensator):
    """Low-rank linear compensator + 残差连接。

    forward = x + up(down(x))，初始 up 权重全零，即 up(down(x))=0，起点严格等于 plain。

    原来没有残差连接且 up 用 kaiming 初始化是错的：
      - 低秩矩阵（rank << channels）在数学上不可能表达恒等映射
      - kaiming 初始化使得初始输出是大幅度随机噪声，直接导致梯度爆炸崩溃
    改成 zero-init(up) + 残差后，初始输出 = x，后续只优化 delta。
    """

    def __init__(self, channels: int, rank: int = 16, activation: str = "gelu") -> None:
        super().__init__()
        rank = max(1, min(rank, channels))
        self.down = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.up   = nn.Conv2d(rank, channels, kernel_size=1, bias=False)
        # down：kaiming 均匀初始化，提供多样的投影方向
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        # up：zero-init，保证初始 delta = up(down(x)) = 0
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.down(x))


class AdapterCompensator(BaseCompensator):
    """非线性上界补偿器：y = x + W_2(σ(W_1(z)))

    forward = x + up(act(down(x)))，初始 up 权重全零，起点严格等于 plain。

    表达力最强（含非线性），作为 Upper Bound 衡量线性补偿器与理论上限的差距。
    由于引入额外非线性计算图，无法通过重参数化融入主干（fusible=False），
    部署时会有真实的计算开销。

    原来 up 用 kaiming 初始化且没有残差连接，与 LoRA 有相同问题，同样改掉。
    """

    fusible = False

    def __init__(self, channels: int, rank: int = 16, activation: str = "gelu") -> None:
        super().__init__()
        rank = max(1, min(rank, channels))
        self.down = nn.Conv2d(channels, rank, kernel_size=1, bias=True)
        self.up   = nn.Conv2d(rank, channels, kernel_size=1, bias=True)
        self.act  = _build_activation(activation)

        # down：kaiming 初始化，bias 初始化为小值避免 dead init
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        if self.down.bias is not None:
            nn.init.uniform_(self.down.bias, -0.01, 0.01)
        # up：zero-init，保证初始 delta = up(act(down(x))) = 0
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


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