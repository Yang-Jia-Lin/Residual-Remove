from __future__ import annotations

import torch
from torch import nn

from src.evaluation.accuracy import extract_logits, topk_accuracy


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    mode: str = "full",
    removed_blocks: list[str] | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device)
            targets = targets.to(device)
            logits = extract_logits(model(images, mode=mode, removed_blocks=removed_blocks))
            loss = criterion(logits, targets)
            top1, top5 = topk_accuracy(logits, targets, topk=(1, min(5, logits.size(1))))
            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_top1 += float(top1.item()) * batch_size
            total_top5 += float(top5.item()) * batch_size
            total_examples += batch_size

    if total_examples == 0:
        return {"loss": 0.0, "top1": 0.0, "top5": 0.0}
    return {
        "loss": total_loss / total_examples,
        "top1": total_top1 / total_examples,
        "top5": total_top5 / total_examples,
    }
