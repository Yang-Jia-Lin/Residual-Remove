"""Src/Models_Evaluation/accuracy.py"""

import time
from collections.abc import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from Src.Utils.runtime import extract_logits


def topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Iterable[int] = (1, 5),
) -> list[torch.Tensor]:
    """计算 Top-K 准确率，返回百分比（0~100）"""
    topk = list(topk)
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


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mode: str = "full",
    removed_blocks: list[str] | None = None,
    max_batches: int | None = None,
):
    """在给定 dataloader 上评估模型，返回 loss / top1 / top5。"""
    # 延迟导入以避免和 trainer 形成模块级循环依赖。
    from Src.Training.trainer import EpochResult, _forward

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
            top1, top5 = topk_accuracy(
                logits, targets, topk=(1, min(5, logits.size(1)))
            )

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
