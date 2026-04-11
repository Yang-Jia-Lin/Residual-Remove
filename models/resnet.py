from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn

from .compensators import build_compensator, freeze_backbone_except_compensators, trainable_compensator_parameters


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        block_name: str = "",
        compensator_name: str = "identity",
        compensator_rank: int = 16,
        adapter_activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.block_name = block_name
        self.compensator = build_compensator(
            compensator_name,
            channels=planes * self.expansion,
            rank=compensator_rank,
            activation=adapter_activation,
        )

    def _compute_plain_and_identity(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out, identity

    def forward_collect(self, x: torch.Tensor, mode: str = "full") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        plain, identity = self._compute_plain_and_identity(x)
        if mode == "full":
            out = plain + identity
        elif mode == "plain":
            out = plain
        elif mode == "compensated":
            out = self.compensator(plain)
        else:
            raise ValueError(f"Unsupported block mode: {mode}")
        out = self.relu(out)
        return out, {"plain": plain, "identity": identity, "output": out}

    def forward(self, x: torch.Tensor, mode: str = "full") -> torch.Tensor:
        out, _ = self.forward_collect(x, mode=mode)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        block_name: str = "",
        compensator_name: str = "identity",
        compensator_rank: int = 16,
        adapter_activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.block_name = block_name
        self.compensator = build_compensator(
            compensator_name,
            channels=planes * self.expansion,
            rank=compensator_rank,
            activation=adapter_activation,
        )

    def _compute_plain_and_identity(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        return out, identity

    def forward_collect(self, x: torch.Tensor, mode: str = "full") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        plain, identity = self._compute_plain_and_identity(x)
        if mode == "full":
            out = plain + identity
        elif mode == "plain":
            out = plain
        elif mode == "compensated":
            out = self.compensator(plain)
        else:
            raise ValueError(f"Unsupported block mode: {mode}")
        out = self.relu(out)
        return out, {"plain": plain, "identity": identity, "output": out}

    def forward(self, x: torch.Tensor, mode: str = "full") -> torch.Tensor:
        out, _ = self.forward_collect(x, mode=mode)
        return out


class ResidualResNet(nn.Module):
    def __init__(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        layers: list[int],
        num_classes: int = 1000,
        small_input: bool = False,
        compensator_name: str = "identity",
        compensator_rank: int = 16,
        adapter_activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.compensator_name = compensator_name
        self.compensator_rank = compensator_rank
        self.adapter_activation = adapter_activation

        if small_input:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.block_order: list[str] = []
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, stage_name="layer1")
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, stage_name="layer2")
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, stage_name="layer3")
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, stage_name="layer4")

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                nn.init.zeros_(module.bias)

    def _make_layer(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int,
        stage_name: str,
    ) -> nn.ModuleList:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = nn.ModuleList()
        first_name = f"{stage_name}.0"
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                block_name=first_name,
                compensator_name=self.compensator_name,
                compensator_rank=self.compensator_rank,
                adapter_activation=self.adapter_activation,
            )
        )
        self.block_order.append(first_name)
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            block_name = f"{stage_name}.{index}"
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    downsample=None,
                    block_name=block_name,
                    compensator_name=self.compensator_name,
                    compensator_rank=self.compensator_rank,
                    adapter_activation=self.adapter_activation,
                )
            )
            self.block_order.append(block_name)
        return layers

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

    def _resolve_block_mode(self, block_name: str, mode: str, removed_blocks: set[str]) -> str:
        if mode == "full":
            return "full"
        return mode if block_name in removed_blocks else "full"

    def _forward_stem(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def _forward_blocks(
        self,
        x: torch.Tensor,
        mode: str,
        removed_blocks: set[str],
        return_features: bool = False,
        return_residual_stats: bool = False,
        stop_after: str | None = None,
        start_after: str | None = None,
    ) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor], OrderedDict[str, dict[str, torch.Tensor]]]:
        features: OrderedDict[str, torch.Tensor] = OrderedDict()
        residual_stats: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        started = start_after is None
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in stage:
                if not started:
                    if block.block_name == start_after:
                        started = True
                    continue
                block_mode = self._resolve_block_mode(block.block_name, mode, removed_blocks)
                if return_residual_stats:
                    x, stats = block.forward_collect(x, mode=block_mode)
                    residual_stats[block.block_name] = stats
                else:
                    x = block(x, mode=block_mode)
                if return_features:
                    features[block.block_name] = x
                if stop_after is not None and block.block_name == stop_after:
                    return x, features, residual_stats
        return x, features, residual_stats

    def _forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "full",
        removed_blocks: list[str] | set[str] | None = None,
        return_features: bool = False,
        return_residual_stats: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        removed = self._normalize_removed_blocks(mode, removed_blocks)
        x = self._forward_stem(x)
        features = OrderedDict()
        if return_features:
            features["stem"] = x
        x, block_features, residual_stats = self._forward_blocks(
            x,
            mode=mode,
            removed_blocks=removed,
            return_features=return_features,
            return_residual_stats=return_residual_stats,
        )
        if return_features:
            features.update(block_features)
        logits = self._forward_head(x)
        if not return_features and not return_residual_stats:
            return logits
        return {"logits": logits, "features": features, "residual_stats": residual_stats}

    def get_split_points(self) -> list[str]:
        return ["stem", *self.block_order]

    def forward_to_split(
        self,
        x: torch.Tensor,
        split_point: str,
        mode: str = "full",
        removed_blocks: list[str] | set[str] | None = None,
    ) -> torch.Tensor:
        removed = self._normalize_removed_blocks(mode, removed_blocks)
        x = self._forward_stem(x)
        if split_point == "stem":
            return x
        x, _, _ = self._forward_blocks(x, mode=mode, removed_blocks=removed, stop_after=split_point)
        return x

    def forward_from_split(
        self,
        x: torch.Tensor,
        split_point: str,
        mode: str = "full",
        removed_blocks: list[str] | set[str] | None = None,
    ) -> torch.Tensor:
        removed = self._normalize_removed_blocks(mode, removed_blocks)
        if split_point == "stem":
            x, _, _ = self._forward_blocks(x, mode=mode, removed_blocks=removed)
        else:
            x, _, _ = self._forward_blocks(x, mode=mode, removed_blocks=removed, start_after=split_point)
        return self._forward_head(x)


def build_resnet(
    depth: int,
    num_classes: int = 1000,
    input_size: int = 224,
    compensator_name: str = "identity",
    compensator_rank: int = 16,
    adapter_activation: str = "gelu",
) -> ResidualResNet:
    if depth == 18:
        block: type[BasicBlock] | type[Bottleneck] = BasicBlock
        layers = [2, 2, 2, 2]
    elif depth == 34:
        block = BasicBlock
        layers = [3, 4, 6, 3]
    elif depth == 50:
        block = Bottleneck
        layers = [3, 4, 6, 3]
    else:
        raise ValueError(f"Unsupported ResNet depth: {depth}")
    return ResidualResNet(
        block=block,
        layers=layers,
        num_classes=num_classes,
        small_input=input_size <= 64,
        compensator_name=compensator_name,
        compensator_rank=compensator_rank,
        adapter_activation=adapter_activation,
    )
