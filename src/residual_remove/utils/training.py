from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from tqdm import tqdm


def _top1_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
    predictions = logits.argmax(dim=1)
    return int((predictions == targets).sum().item())


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_samples = 0

    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        running_correct += _top1_correct(outputs, labels)
        running_samples += batch_size

        progress.set_postfix(
            loss=running_loss / max(1, running_samples),
            acc=running_correct / max(1, running_samples),
        )

    return {
        "train_loss": running_loss / max(1, running_samples),
        "train_top1": running_correct / max(1, running_samples),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_samples = 0

    progress = tqdm(loader, desc="eval", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        running_correct += _top1_correct(outputs, labels)
        running_samples += batch_size

        progress.set_postfix(
            loss=running_loss / max(1, running_samples),
            acc=running_correct / max(1, running_samples),
        )

    return {
        "val_loss": running_loss / max(1, running_samples),
        "val_top1": running_correct / max(1, running_samples),
    }
