from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn

from .compensators import build_compensator, freeze_backbone_except_compensators, trainable_compensator_parameters


def _make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor, int((value + divisor / 2) // divisor * divisor))


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation_layer: type[nn.Module] = nn.ReLU6,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            activation_layer(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
        block_name: str,
        compensator_name: str = "identity",
        compensator_rank: int = 16,
        adapter_activation: str = "gelu",
    ) -> None:
        super().__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block_name = block_name

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1))
        layers.extend(
            [
                ConvBNAct(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.compensator = build_compensator(
            compensator_name,
            channels=out_channels,
            rank=compensator_rank,
            activation=adapter_activation,
        )

    def forward_collect(self, x: torch.Tensor, mode: str = "full") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        plain = self.block(x)
        identity = x
        if self.use_residual:
            if mode == "full":
                out = plain + identity
            elif mode == "plain":
                out = plain
            elif mode == "compensated":
                out = self.compensator(plain)
            else:
                raise ValueError(f"Unsupported block mode: {mode}")
            return out, {"plain": plain, "identity": identity, "output": out}
        return plain, {"plain": plain, "identity": identity, "output": plain}

    def forward(self, x: torch.Tensor, mode: str = "full") -> torch.Tensor:
        out, _ = self.forward_collect(x, mode=mode)
        return out


class ResidualMobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        input_size: int = 224,
        compensator_name: str = "identity",
        compensator_rank: int = 16,
        adapter_activation: str = "gelu",
    ) -> None:
        super().__init__()
        input_channel = _make_divisible(32 * width_mult)
        last_channel = _make_divisible(1280 * max(1.0, width_mult))
        self.stem = ConvBNAct(3, input_channel, stride=1 if input_size <= 64 else 2)

        setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.features = nn.ModuleList()
        self.block_order: list[str] = []
        index = 0
        for expand_ratio, channels, repeats, stride in setting:
            output_channel = _make_divisible(channels * width_mult)
            for repeat_idx in range(repeats):
                block_stride = stride if repeat_idx == 0 else 1
                block_name = f"features.{index}"
                block = InvertedResidual(
                    input_channel,
                    output_channel,
                    stride=block_stride,
                    expand_ratio=expand_ratio,
                    block_name=block_name,
                    compensator_name=compensator_name,
                    compensator_rank=compensator_rank,
                    adapter_activation=adapter_activation,
                )
                self.features.append(block)
                if block.use_residual:
                    self.block_order.append(block_name)
                input_channel = output_channel
                index += 1

        self.head = ConvBNAct(input_channel, last_channel, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(last_channel, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                nn.init.zeros_(module.bias)

    def get_block_names(self) -> list[str]:
        return list(self.block_order)

    def freeze_backbone(self) -> None:
        freeze_backbone_except_compensators(self)

    def compensator_parameters(self) -> list[nn.Parameter]:
        return trainable_compensator_parameters(self)

    def _normalize_removed_blocks(self, mode: str, removed_blocks: list[str] | set[str] | None) -> set[str]:
        if mode == "full":
            return set()
        if removed_blocks is None:
            return set(self.block_order)
        return set(removed_blocks)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "full",
        removed_blocks: list[str] | set[str] | None = None,
        return_features: bool = False,
        return_residual_stats: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        removed = self._normalize_removed_blocks(mode, removed_blocks)
        features = OrderedDict()
        residual_stats = OrderedDict()

        x = self.stem(x)
        if return_features:
            features["stem"] = x

        for block in self.features:
            block_mode = "full"
            if block.use_residual and mode != "full" and block.block_name in removed:
                block_mode = mode
            if return_residual_stats:
                x, stats = block.forward_collect(x, mode=block_mode)
                if block.use_residual:
                    residual_stats[block.block_name] = stats
            else:
                x = block(x, mode=block_mode)
            if return_features and block.use_residual:
                features[block.block_name] = x

        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        if not return_features and not return_residual_stats:
            return logits
        return {"logits": logits, "features": features, "residual_stats": residual_stats}
