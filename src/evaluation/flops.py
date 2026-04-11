from __future__ import annotations

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def estimate_macs(model: nn.Module, sample: torch.Tensor, **forward_kwargs) -> int:
    hooks = []
    total_macs = 0

    def conv_hook(module: nn.Conv2d, inputs, output) -> None:
        nonlocal total_macs
        batch_size = output.shape[0]
        out_h, out_w = output.shape[2], output.shape[3]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        total_macs += batch_size * output.shape[1] * out_h * out_w * kernel_ops

    def linear_hook(module: nn.Linear, inputs, output) -> None:
        nonlocal total_macs
        batch_size = output.shape[0]
        total_macs += batch_size * module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(sample, **forward_kwargs)
    if was_training:
        model.train()
    for hook in hooks:
        hook.remove()
    return int(total_macs)
