"""Src/DatasetsProcess/split_imagenet100.py"""

"""
split_imagenet100.py

对 ImageNet100 做固定的训练/验证集切分，切分结果保存为索引文件，
后续每次加载数据集直接读索引，保证可复现。

用法：
    python split_imagenet100.py              # 生成切分索引
    python split_imagenet100.py --verify     # 验证切分结果
"""

import os
import json
import argparse
import torch
from torchvision import datasets, transforms


# ─── 配置 ────────────────────────────────────────────────────────────────────
DATA_ROOT   = "/root/autodl-tmp/0-Data/ImageNet100/imagenet100"
SPLIT_FILE  = "/root/autodl-tmp/0-Data/ImageNet100/split_indices.json"
VAL_RATIO   = 0.2
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────


def generate_split():
    print(f"[1/3] 加载数据集：{DATA_ROOT}")

    # 只扫描目录结构，不实际读图片，速度很快
    dataset = datasets.ImageFolder(root=DATA_ROOT, transform=transforms.ToTensor())
    total   = len(dataset)
    n_cls   = len(dataset.classes)
    print(f"      共 {total} 张图，{n_cls} 个类别")

    print(f"[2/3] 按 {int((1-VAL_RATIO)*100)}/{int(VAL_RATIO*100)} 切分，seed={SEED}")
    val_size   = int(VAL_RATIO * total)
    train_size = total - val_size

    train_subset, val_subset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    split = {
        "seed":        SEED,
        "val_ratio":   VAL_RATIO,
        "total":       total,
        "train_size":  train_size,
        "val_size":    val_size,
        "train_indices": train_subset.indices,
        "val_indices":   val_subset.indices,
    }

    print(f"[3/3] 保存索引到：{SPLIT_FILE}")
    os.makedirs(os.path.dirname(SPLIT_FILE), exist_ok=True)
    with open(SPLIT_FILE, "w") as f:
        json.dump(split, f)

    print(f"\n✅ 完成：train={train_size}，val={val_size}")


def verify_split():
    print(f"读取索引文件：{SPLIT_FILE}")
    with open(SPLIT_FILE, "r") as f:
        split = json.load(f)

    print(f"  seed       = {split['seed']}")
    print(f"  total      = {split['total']}")
    print(f"  train_size = {split['train_size']}")
    print(f"  val_size   = {split['val_size']}")

    # 检查有没有交集（不应该有）
    train_set = set(split["train_indices"])
    val_set   = set(split["val_indices"])
    overlap   = train_set & val_set
    print(f"  train/val 索引交集：{len(overlap)}（应为 0）")

    if len(overlap) == 0:
        print("\n✅ 切分干净，无数据泄漏")
    else:
        print("\n❌ 存在重叠，请重新生成！")


# ─── 供其他模块 import 使用 ──────────────────────────────────────────────────
def get_datasets(train_transform=None, val_transform=None):
    """
    返回固定切分的 train/val Dataset，直接在训练代码里 import 用。

    示例：
        from split_imagenet100 import get_datasets
        train_ds, val_ds = get_datasets(train_transform=..., val_transform=...)
    """
    with open(SPLIT_FILE, "r") as f:
        split = json.load(f)

    # 注意：train 和 val 各自加载一次，这样 transform 互不干扰
    base_train = datasets.ImageFolder(root=DATA_ROOT, transform=train_transform)
    base_val   = datasets.ImageFolder(root=DATA_ROOT, transform=val_transform)

    train_dataset = torch.utils.data.Subset(base_train, split["train_indices"])
    val_dataset   = torch.utils.data.Subset(base_val,   split["val_indices"])

    return train_dataset, val_dataset
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="验证已有切分文件")
    args = parser.parse_args()

    if args.verify:
        verify_split()
    else:
        generate_split()