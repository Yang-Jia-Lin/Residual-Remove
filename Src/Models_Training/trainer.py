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

from Src.Models_Evaluation.accuracy import extract_logits, topk_accuracy


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


# ── 评估 ──────────────────────────────
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mode: str = "full",
    removed_blocks: list[str] | None = None,
    max_batches: int | None = None,
) -> EpochResult:
    """在给定 dataloader 上评估模型，返回 loss / top1 / top5。

    注意返回类型从 dict 改成了 EpochResult dataclass，
    使用时用 result.top1 而不是 result["top1"]，更安全也更易补全。
    """
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

            images  = images.to(device)
            targets = targets.to(device)
            logits  = extract_logits(_forward(model, images, mode, removed_blocks))
            loss    = criterion(logits, targets)
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


if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from torch import optim

    from Src.Utils.datasets import make_dataloaders
    from Src.Models_Nets import build_model

    # ── 第一步：加载配置 ─────────────────────────────────────────────────────
    # 和 datasets.py 一样，yaml 只在入口处读取一次，
    # 具体的值以普通参数形式向下传递，trainer 本身不感知 yaml 的存在。
    cfg_path = Path("_configs/default_env.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    device = torch.device(cfg["device"])
    seed   = cfg["seed"]
    torch.manual_seed(seed)
    print(f"配置加载自：{cfg_path}")
    print(f"  device : {device}")
    print(f"  seed   : {seed}")

    # ── 第二步：准备数据 ─────────────────────────────────────────────────────
    # 这里故意用合成数据（synthetic_if_missing=True），
    # 这样冒烟测试不依赖任何本地数据集，任何机器上都能跑通。
    # 同时把 train_size 设得很小（64张），只是为了验证流程，不是为了训练出好模型。
    bundle = make_dataloaders(
        dataset_name         = cfg["data"]["default_dataset"],
        data_root            = cfg["paths"]["data_root"],
        batch_size           = 16,
        image_size           = cfg["data"]["default_image_size"],
        num_workers          = cfg["num_workers"],
        synthetic_if_missing = True,   # 冒烟测试强制用合成数据，不依赖真实数据集
        train_size           = 64,     # 只要够跑几个 batch 就行
        val_size             = 32,
        seed                 = seed,
    )
    print(f"\n数据集来源：{bundle.source}（{bundle.num_classes} 类）")

    # ── 第三步：构建模型 ─────────────────────────────────────────────────────
    # 用 ResNet-18 + IdentityCompensator，这是最轻量的组合，冒烟测试够用。
    # pretrained=False 因为这里不需要好的初始权重，只是验证代码路径。
    model = build_model(
        "resnet18",
        num_classes      = bundle.num_classes,
        pretrained       = False,
        compensator_name = "identity",
    ).to(device)
    print(f"模型构建完成：ResNet-18，{bundle.num_classes} 类")

    # ── 第四步：构建优化器和调度器 ───────────────────────────────────────────
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # StepLR 每个 epoch 都会衰减，这里只是为了验证 scheduler 参数能正常传进 train_model
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # ── 第五步：运行完整训练流程（只跑 2 个 epoch）────────────────────────────
    # 2 个 epoch 足够验证：
    #   - train_one_epoch 的前向 + 反向 + 参数更新
    #   - evaluate_model 的推理 + 指标计算
    #   - scheduler.step() 不报错
    #   - checkpoint 保存路径能自动创建
    print("\n开始训练（2 个 epoch，仅用于冒烟测试）...")
    ckpt_path = Path(cfg["paths"]["checkpoint_root"]) / "smoke_test" / "best.pth"

    history = train_model(
        model            = model,
        train_loader     = bundle.train_loader,
        val_loader       = bundle.val_loader,
        optimizer        = optimizer,
        device           = device,
        num_epochs       = 2,
        scheduler        = scheduler,
        mode             = "full",       # 正常带残差的前向，baseline 的标准模式
        checkpoint_path  = ckpt_path,
        verbose          = True,
    )

    # ── 第六步：验证历史记录是否完整 ─────────────────────────────────────────
    # TrainHistory 里应该有 2 个 train EpochResult 和 2 个 val EpochResult。
    # 如果数量不对，说明训练循环在中途被意外打断了。
    assert len(history.train) == 2, f"期望 2 个训练记录，实际 {len(history.train)}"
    assert len(history.val)   == 2, f"期望 2 个验证记录，实际 {len(history.val)}"
    print(f"\n训练历史记录条数正确：train={len(history.train)}, val={len(history.val)}")

    # ── 第七步：验证指标值的合理性 ───────────────────────────────────────────
    # 这里不验证精度高低，只验证"值域合理"：
    #   - loss 必须是有限正数（如果是 nan 或 inf，说明反向传播出了问题）
    #   - top1 必须在 [0, 100] 范围内（按百分比计）
    for i, (tr, va) in enumerate(zip(history.train, history.val)):
        assert torch.isfinite(torch.tensor(tr.loss)), f"第 {i+1} epoch 训练 loss 不是有限值：{tr.loss}"
        assert torch.isfinite(torch.tensor(va.loss)), f"第 {i+1} epoch 验证 loss 不是有限值：{va.loss}"
        assert 0.0 <= tr.top1 <= 100.0, f"训练 top1 超出合理范围：{tr.top1}"
        assert 0.0 <= va.top1 <= 100.0, f"验证 top1 超出合理范围：{va.top1}"
    print("所有 epoch 的 loss 和 top1 值域均合理")

    # ── 第八步：验证 plain mode（动机实验的核心模式）────────────────────────
    # 单独跑一次 evaluate_model，切换到 plain mode，
    # 验证删除残差后的推理路径也能正常走通、不报错。
    print("\n验证 plain mode（删除全部残差）...")
    plain_result = evaluate_model(
        model          = model,
        dataloader     = bundle.val_loader,
        device         = device,
        mode           = "plain",          # 动机实验用这个 mode
        removed_blocks = None,             # None 表示删除所有 block 的残差
        max_batches    = 2,                # 只跑 2 个 batch，够验证路径就行
    )
    print(f"  plain mode 验证 loss  : {plain_result.loss:.4f}")
    print(f"  plain mode 验证 top1  : {plain_result.top1:.2f}%")
    # plain mode 下精度大幅下降是正常的（这正是动机实验想证明的），
    # 这里只验证数值是有限的，不对精度做任何断言。
    assert torch.isfinite(torch.tensor(plain_result.loss)), "plain mode 的 loss 不是有限值"

    # ── 第九步：验证 checkpoint 文件确实被写入磁盘 ──────────────────────────
    # train_model 在 val top1 刷新最优时才保存，
    # 跑了 2 个 epoch，理论上第 1 个 epoch 就会保存一次（因为初始 best=0.0）。
    if ckpt_path.exists():
        size_kb = ckpt_path.stat().st_size / 1024
        print(f"\nCheckpoint 已保存至：{ckpt_path}（{size_kb:.1f} KB）")
    else:
        # 合成数据下随机初始化模型 top1 可能始终为 0，导致没有触发保存
        # 这不算失败，打印警告即可
        print(f"\n⚠ Checkpoint 未生成（val top1 可能始终为 0，属正常现象）")

    print("\n✓ trainer.py 冒烟测试全部通过。")