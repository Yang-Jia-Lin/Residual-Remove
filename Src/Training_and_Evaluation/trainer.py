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

from Src.Metrics.accuracy import extract_logits, topk_accuracy


# ── 训练结果的结构化容器 ────────────────────────────────────────────────────────

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


# ── 核心工具函数 ────────────────────────────────────────────────────────────────

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


# ── 单 epoch 训练 ───────────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    mode: str = "full",
    removed_blocks: list[str] | None = None,
    max_batches: int | None = None,
    # 可选的 batch 级别回调，比如用于打印进度
    on_batch_end: Callable[[int, float], None] | None = None,
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


# ── 完整训练流程 ────────────────────────────────────────────────────────────────

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
    """完整的训练 + 验证循环。

    每个 epoch 结束后自动在 val_loader 上评估，并保存到目前为止
    val top1 最高的 checkpoint（best model checkpoint 策略）。

    Args:
        checkpoint_path: 如果提供，会在每次 val top1 刷新最优时保存权重。
                         文件名本身即为保存路径，例如：
                         "_logs/checkpoints/resnet50_cifar100/best.pth"
    """
    from Src.Training_and_Evaluation.evaluator import evaluate_model

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
