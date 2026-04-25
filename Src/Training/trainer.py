"""Src/Models_Training/trainer.py"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from Src.Metrics.accuracy import evaluate_model, extract_logits, topk_accuracy


# ── 训练结果结构 ───

@dataclass
class EpochResult:
    """单个 epoch 的训练或验证指标快照。"""
    loss: float
    top1: float
    top5: float
    elapsed_seconds: float


@dataclass
class TrainHistory:
    """完整训练过程的指标历史，便于后续画图。"""
    train: list[EpochResult] = field(default_factory=list)
    val:   list[EpochResult] = field(default_factory=list)

    def best_val_top1(self) -> float:
        return max((r.top1 for r in self.val), default=0.0)


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


def _forward(
    model: nn.Module,
    images: torch.Tensor,
    mode: str,
    removed_blocks: list[str] | None,
) -> torch.Tensor:
    # 检查模型是否支持 mode 参数（即是否是 InjectedModel）
    if hasattr(model, "get_block_names"):
        return model(images, mode=mode, removed_blocks=removed_blocks)
    return model(images)


# ── 单 epoch 训练 ───
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    mode: str = "full",
    removed_blocks: list[str] | None = None,
    max_batches: int | None = None,
    on_batch_end: Callable[[int, float], None] | None = None, # 可选的 batch 级别回调，比如用于打印进度
) -> EpochResult:
    """运行一个完整的训练 epoch，返回该 epoch 的平均指标。"""
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_examples = 0
    t_start = time.perf_counter()

    for batch_idx, (images, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images  = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = extract_logits(_forward(model, images, mode, removed_blocks))
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        top1, top5 = topk_accuracy(logits, targets, topk=(1, min(5, logits.size(1))))
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_top1 += float(top1.item()) * batch_size
        total_top5 += float(top5.item()) * batch_size
        total_examples += batch_size

        if on_batch_end is not None:
            on_batch_end(batch_idx, float(loss.item()))

    n = max(total_examples, 1)
    return EpochResult(
        loss=total_loss / n,
        top1=total_top1 / n,
        top5=total_top5 / n,
        elapsed_seconds=time.perf_counter() - t_start,
    )


# ── 完整训练 ───

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    num_epochs: int,
    criterion: nn.Module | None = None,
    scheduler: LRScheduler | None = None,
    mode: str = "full",
    removed_blocks: list[str] | None = None,
    checkpoint_path: Path | str | None = None,
    verbose: bool = True,
) -> TrainHistory:
    """完整的训练 + 验证"""
    criterion = criterion or nn.CrossEntropyLoss()
    history = TrainHistory()
    best_val_top1 = 0.0

    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        # ── 训练 ──
        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            mode=mode, removed_blocks=removed_blocks,
        )
        history.train.append(train_result)

        # ── 验证 ──
        val_result = evaluate_model(
            model, val_loader, device,
            mode=mode, removed_blocks=removed_blocks,
        )
        history.val.append(val_result)

        # ── 学习率调度 ──
        if scheduler is not None:
            scheduler.step()

        # ── 保存最优 checkpoint ──
        if val_result.top1 > best_val_top1 and checkpoint_path is not None:
            best_val_top1 = val_result.top1
            torch.save(model.state_dict(), checkpoint_path)

        if verbose:
            print(
                f"Epoch {epoch:03d}/{num_epochs} │ "
                f"train loss {train_result.loss:.4f}  top1 {train_result.top1:.2f}% │ "
                f"val loss {val_result.loss:.4f}  top1 {val_result.top1:.2f}%  "
                f"({'✓ saved' if val_result.top1 == best_val_top1 else ''})"
            )

    return history
