"""把官方模型中匹配的 block 替换成 PatchedBlock，返回 InjectedModel"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import torch
from torch import nn

from Src.Models_Nets.compensators import BaseCompensator, build_compensator, freeze_backbone_except_compensators, trainable_compensator_parameters
from Src.Models_Nets.origin.mobilenet import InvertedResidual
from Src.Models_Nets.origin.resnet import BasicBlock, Bottleneck


@dataclass(frozen=True)
class BlockSpec:
    """模块类型：说明如何提取其补偿器通道数以及如何包装成 PatchedBlock"""
    block_class: type[nn.Module]
    get_channels: Callable[[nn.Module], int]
    wrap: Callable[[nn.Module, str, BaseCompensator], "PatchedBlock"]


def _postprocess_output(original_block: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """对残差分支输出进行后处理(补充ReLU)，确保与原始块的输出一致。"""
    if isinstance(original_block, (BasicBlock, Bottleneck)):
        return original_block.relu(tensor)
    return tensor


def _split_forward(original_block: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """返回加法前的残差分支输出和恒等分支输出 F(x, W)"""

    if isinstance(original_block, BasicBlock):
        identity = x
        out = original_block.conv1(x)
        out = original_block.bn1(out)
        out = original_block.relu(out)
        out = original_block.conv2(out)
        out = original_block.bn2(out)
        if original_block.downsample is not None:
            identity = original_block.downsample(x)
        return out, identity

    if isinstance(original_block, Bottleneck):
        identity = x
        out = original_block.conv1(x)
        out = original_block.bn1(out)
        out = original_block.relu(out)
        out = original_block.conv2(out)
        out = original_block.bn2(out)
        out = original_block.relu(out)
        out = original_block.conv3(out)
        out = original_block.bn3(out)
        if original_block.downsample is not None:
            identity = original_block.downsample(x)
        return out, identity

    if isinstance(original_block, InvertedResidual):
        return original_block.conv(x), x

    raise TypeError(f"Unsupported block type: {type(original_block).__name__}")


@dataclass(frozen=True)
class _ExecutionPlan:
    stem_modules: list[str]
    body_modules: list[str]
    head_modules: list[str]
    tracked_blocks: list[str]
    architecture: str


class PatchedBlock(nn.Module):
    """封装已打补丁的块及其补偿器的通用封装类"""

    def __init__(
        self,
        original_block: nn.Module,
        compensator: BaseCompensator,
        block_name: str,
        track_in_block_order: bool = True,
    ) -> None:
        super().__init__()
        self.original_block = original_block
        self.compensator = compensator
        self.block_name = block_name
        self.track_in_block_order = track_in_block_order

    # def forward(self, x: torch.Tensor, mode: str = "full") -> torch.Tensor:
    #     if mode == "full":
    #         return self.original_block(x)

    #     plain, _ = _split_forward(self.original_block, x)
    #     if mode == "plain":
    #         return _postprocess_output(self.original_block, plain)
    #     if mode == "compensated":
    #         return _postprocess_output(self.original_block, self.compensator(plain))
    #     raise ValueError(f"Unsupported block mode: {mode}")

    def forward(self, x: torch.Tensor, mode: str = "full") -> torch.Tensor:
        if mode == "full":
            return self.original_block(x)

        # 核心修复：如果是 plain 或 compensated，彻底只算主分支
        if mode == "plain":
            plain = _forward_plain(self.original_block, x)
            return _postprocess_output(self.original_block, plain)
            
        if mode == "compensated":
            plain = _forward_plain(self.original_block, x)
            return _postprocess_output(self.original_block, self.compensator(plain))
            
        raise ValueError(f"Unsupported block mode: {mode}")
    
    def forward_collect(self, x: torch.Tensor, mode: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        plain, identity = _split_forward(self.original_block, x)
        output = self.forward(x, mode=mode)
        return output, {"plain": plain, "identity": identity, "output": output}


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, new_module)


def _match_block_spec(module: nn.Module, block_specs: Sequence[BlockSpec]) -> BlockSpec | None:
    for spec in block_specs:
        if isinstance(module, spec.block_class):
            return spec
    return None


def _build_execution_plan(model: nn.Module, block_order: list[str]) -> _ExecutionPlan:
    if hasattr(model, "conv1") and hasattr(model, "layer4"):
        return _ExecutionPlan(
            stem_modules=["conv1", "bn1", "relu", "maxpool"],
            body_modules=list(block_order),
            head_modules=["avgpool", "fc"],
            tracked_blocks=list(block_order),
            architecture="resnet",
        )

    if hasattr(model, "features") and hasattr(model, "classifier"):
        feature_names = list(model.features._modules.keys())
        first_tracked_index = len(feature_names)
        tracked_set = set(block_order)
        for index, name in enumerate(feature_names):
            full_name = f"features.{name}"
            if full_name in tracked_set:
                first_tracked_index = index
                break
        stem_modules = [f"features.{name}" for name in feature_names[:first_tracked_index]]
        body_modules = [f"features.{name}" for name in feature_names[first_tracked_index:]]
        return _ExecutionPlan(
            stem_modules=stem_modules,
            body_modules=body_modules,
            head_modules=["classifier"],
            tracked_blocks=list(block_order),
            architecture="mobilenet_v2",
        )

    raise TypeError(f"Unsupported backbone architecture: {type(model).__name__}")


class InjectedModel(nn.Module):
    """提供前向传播方法，允许在不同模式下运行块，并选择性地返回特征和残差统计数据"""

    def __init__(self, backbone: nn.Module, block_order: list[str]) -> None:
        super().__init__()
        self.backbone = backbone
        self.block_order = list(block_order)
        self._plan = _build_execution_plan(backbone, self.block_order)

    def _normalize_removed_blocks(self, mode: str, removed_blocks: list[str] | set[str] | None) -> set[str]:
        if mode == "full":
            return set()
        if removed_blocks is None:
            return set(self.block_order)
        return set(removed_blocks)

    def _run_modules(self, x: torch.Tensor, module_names: list[str]) -> torch.Tensor:
        for module_name in module_names:
            module = self.backbone.get_submodule(module_name)
            x = module(x)
        return x

    def _run_blocks(
        self,
        x: torch.Tensor,
        mode: str,
        removed_blocks: set[str],
        return_features: bool = False,
        return_residual_stats: bool = False,
        stop_after: str | None = None,
        start_after: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        features: dict[str, torch.Tensor] = {}
        residual_stats: dict[str, dict[str, torch.Tensor]] = {}
        started = start_after is None

        for module_name in self._plan.body_modules:
            if not started:
                if module_name == start_after:
                    started = True
                continue

            module = self.backbone.get_submodule(module_name)
            if isinstance(module, PatchedBlock):
                block_mode = mode if module_name in removed_blocks else "full"
                if return_residual_stats:
                    x, stats = module.forward_collect(x, mode=block_mode)
                    if module.track_in_block_order:
                        residual_stats[module_name] = stats
                else:
                    x = module(x, mode=block_mode)
                if return_features and module.track_in_block_order:
                    features[module_name] = x
            else:
                x = module(x)

            if stop_after is not None and module_name == stop_after:
                break

        return x, features, residual_stats

    def _forward_head(self, x: torch.Tensor) -> torch.Tensor:
        if self._plan.architecture == "resnet":
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            return self.backbone.fc(x)

        if self._plan.architecture == "mobilenet_v2":
            x = nn.functional.adaptive_avg_pool2d(x, 1)
            x = torch.flatten(x, 1)
            return self.backbone.classifier(x)

        raise TypeError(f"Unsupported architecture: {self._plan.architecture}")

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "full",
        removed_blocks: list[str] | set[str] | None = None,
        return_features: bool = False,
        return_residual_stats: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        removed = self._normalize_removed_blocks(mode, removed_blocks)
        x = self._run_modules(x, self._plan.stem_modules)

        features: dict[str, torch.Tensor] = {}
        residual_stats: dict[str, dict[str, torch.Tensor]] = {}
        if return_features:
            features["stem"] = x

        x, block_features, block_stats = self._run_blocks(
            x,
            mode=mode,
            removed_blocks=removed,
            return_features=return_features,
            return_residual_stats=return_residual_stats,
        )
        if return_features:
            features.update(block_features)
        if return_residual_stats:
            residual_stats.update(block_stats)

        logits = self._forward_head(x)
        if not return_features and not return_residual_stats:
            return logits
        return {"logits": logits, "features": features, "residual_stats": residual_stats}

    def forward_to_split(
        self,
        x: torch.Tensor,
        split_point: str,
        mode: str = "full",
        removed_blocks: list[str] | set[str] | None = None,
    ) -> torch.Tensor:
        removed = self._normalize_removed_blocks(mode, removed_blocks)
        if split_point == "stem":
            return self._run_modules(x, self._plan.stem_modules)

        x = self._run_modules(x, self._plan.stem_modules)
        x, _, _ = self._run_blocks(x, mode=mode, removed_blocks=removed, stop_after=split_point)
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
            x, _, _ = self._run_blocks(x, mode=mode, removed_blocks=removed)
            return self._forward_head(x)

        x, _, _ = self._run_blocks(x, mode=mode, removed_blocks=removed, start_after=split_point)
        return self._forward_head(x)

    def freeze_backbone(self) -> None:
        freeze_backbone_except_compensators(self)

    def compensator_parameters(self) -> list[nn.Parameter]:
        return trainable_compensator_parameters(self)

    def get_block_names(self) -> list[str]:
        return list(self.block_order)

    def get_split_points(self) -> list[str]:
        return ["stem", *self.block_order]


def _default_wrap(original: nn.Module, block_name: str, compensator: BaseCompensator) -> PatchedBlock:
    track_in_order = True
    if hasattr(original, "use_res_connect"):
        track_in_order = bool(getattr(original, "use_res_connect"))
    return PatchedBlock(original_block=original, compensator=compensator, block_name=block_name, track_in_block_order=track_in_order)


def _resnet_basic_channels(module: nn.Module) -> int:
    return int(module.bn2.num_features)


def _resnet_bottleneck_channels(module: nn.Module) -> int:
    return int(module.bn3.num_features)


def _mobilenet_channels(module: nn.Module) -> int:
    return int(module.conv[-1].num_features)


resnet_block_specs = (
    BlockSpec(block_class=BasicBlock, get_channels=_resnet_basic_channels, wrap=_default_wrap),
    BlockSpec(block_class=Bottleneck, get_channels=_resnet_bottleneck_channels, wrap=_default_wrap),
)

mobilenet_block_specs = (
    BlockSpec(block_class=InvertedResidual, get_channels=_mobilenet_channels, wrap=_default_wrap),
)


def inject(
    model: nn.Module,
    block_specs: Sequence[BlockSpec],
    compensator_name: str,
    rank: int,
    activation: str,
) -> InjectedModel:
    """Inject compensator-aware wrappers into an official backbone."""

    matches: list[tuple[str, nn.Module, BlockSpec]] = []
    for module_name, module in model.named_modules():
        if module_name == "":
            continue
        spec = _match_block_spec(module, block_specs)
        if spec is not None:
            matches.append((module_name, module, spec))

    block_order: list[str] = []
    for module_name, module, spec in matches:
        channels = spec.get_channels(module)
        compensator = build_compensator(
            compensator_name,
            channels=channels,
            rank=rank,
            activation=activation,
        )
        patched = spec.wrap(module, module_name, compensator)
        _replace_module(model, module_name, patched)
        if patched.track_in_block_order:
            block_order.append(module_name)

    return InjectedModel(model, block_order)


def _forward_plain(original_block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """仅执行主干特征提取，彻底跳过残差与下采样计算，避免显存峰值"""
    if isinstance(original_block, BasicBlock):
        out = original_block.conv1(x)
        out = original_block.bn1(out)
        out = original_block.relu(out)
        out = original_block.conv2(out)
        out = original_block.bn2(out)
        return out

    if isinstance(original_block, Bottleneck):
        out = original_block.conv1(x)
        out = original_block.bn1(out)
        out = original_block.relu(out)
        out = original_block.conv2(out)
        out = original_block.bn2(out)
        out = original_block.relu(out)
        out = original_block.conv3(out)
        out = original_block.bn3(out)
        return out

    if isinstance(original_block, InvertedResidual):
        return original_block.conv(x)

    raise TypeError(f"Unsupported block type: {type(original_block).__name__}")