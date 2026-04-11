from __future__ import annotations

from collections.abc import Iterable

import torch


def extract_logits(output: torch.Tensor | dict) -> torch.Tensor:
    if isinstance(output, dict):
        return output["logits"]
    return output


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: Iterable[int] = (1, 5)) -> list[torch.Tensor]:
    max_k = max(topk)
    batch_size = targets.size(0)
    _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
    predictions = predictions.t()
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))

    results: list[torch.Tensor] = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append(correct_k.mul_(100.0 / batch_size))
    return results
