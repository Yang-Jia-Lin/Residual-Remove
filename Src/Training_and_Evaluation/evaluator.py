"""Src/Models_Training/evaluater.py"""
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from Src.Metrics.accuracy import extract_logits, topk_accuracy
from Src.Training_and_Evaluation.trainer import EpochResult, _forward


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mode: str = "full",
    removed_blocks: list[str] | None = None,
    max_batches: int | None = None,
) -> EpochResult:
    """在给定 dataloader 上评估模型，返回 loss / top1 / top5。"""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_examples = 0
    t_start = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            targets = targets.to(device)
            logits = extract_logits(_forward(model, images, mode, removed_blocks))
            loss = criterion(logits, targets)
            top1, top5 = topk_accuracy(logits, targets, topk=(1, min(5, logits.size(1))))

            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_top1 += float(top1.item()) * batch_size
            total_top5 += float(top5.item()) * batch_size
            total_examples += batch_size

    n = max(total_examples, 1)
    return EpochResult(
        loss=total_loss / n,
        top1=total_top1 / n,
        top5=total_top5 / n,
        elapsed_seconds=time.perf_counter() - t_start,
    )