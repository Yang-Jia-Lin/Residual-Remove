from __future__ import annotations

import torch
from torch import nn


def feature_mse_loss(
    student_features: dict[str, torch.Tensor],
    teacher_features: dict[str, torch.Tensor],
    layers: list[str],
) -> torch.Tensor:
    criterion = nn.MSELoss()
    losses = []
    for layer in layers:
        if layer in student_features and layer in teacher_features:
            losses.append(criterion(student_features[layer], teacher_features[layer]))
    if not losses:
        reference = next(iter(student_features.values()))
        return reference.new_zeros(())
    return torch.stack(losses).mean()


def logit_mse_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(student_logits, teacher_logits)
