"""Src/Models_Training/finetune.py"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Configs.paras import RESULT_DIR


def finetune_head(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    save_name: str = "finetuned.pth",
) -> nn.Module:
    """冻结主干，只训练 fc 层，完成后保存 checkpoint。"""

    # 冻结所有参数，再单独放开 fc
    for param in model.parameters():
        param.requires_grad = False
    for param in model.backbone.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    best_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        # ── 训练 ──────────────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)

        # ── 验证 ──────────────────────────────────────────────────────
        val_acc = _evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{epochs}  "
            f"loss={total_loss / total:.4f}  val_acc={val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 保存最优 checkpoint
    ckpt_dir = RESULT_DIR / "Checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / save_name
    torch.save({"state_dict": best_state, "best_acc": best_acc}, ckpt_path)
    print(f"\n✅ 最优 val_acc={best_acc:.2f}%，checkpoint 已保存：{ckpt_path}")

    model.load_state_dict(best_state)  # type: ignore
    return model


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def load_finetuned(model: nn.Module, save_name: str = "finetuned.pth") -> nn.Module:
    """加载已保存的 checkpoint，供实验脚本直接使用。"""
    ckpt_path = RESULT_DIR / "Checkpoints" / save_name
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"找不到 checkpoint：{ckpt_path}\n"
            "请先运行 Scripts/Utils/run_finetune.py 生成基线模型。"
        )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    print(f"[load] 已加载 checkpoint（best_acc={ckpt['best_acc']:.2f}%）：{ckpt_path}")
    return model
