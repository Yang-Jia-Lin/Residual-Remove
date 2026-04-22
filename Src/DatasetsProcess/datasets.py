"""Src/DatasetsProcess/datasets.py"""

"""数据集加载与 DataLoader 创建"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
import torch
from torch.utils.data import DataLoader, Dataset, Subset


try:
    from torchvision import datasets as tv_datasets
    from torchvision import transforms as tv_transforms
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False


# ── 返回值容器 ────────────────────────────────────────────────────────────────
@dataclass
class DatasetBundle:
    """make_dataloaders() 的返回值，供实验脚本直接使用。"""
    train_loader: DataLoader
    val_loader:   DataLoader
    train_dataset: Dataset       # <--- 新增
    val_dataset:   Dataset
    num_classes:  int
    input_size:   int
    source:       str   # "cifar10" | "cifar100" | "imagenet" | "synthetic"


# ── 各数据集的归一化统计量 ────────────────────────────────────────────────────────
# 这些值是在各数据集训练集全量上计算得到的逐通道 mean/std。
# 使用预训练权重时，输入的归一化方式必须与训练时保持一致。
_NORM_STATS: dict[str, tuple[list[float], list[float]]] = {
    "cifar10":   ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    "cifar100":  ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    "imagenet":  ([0.485,  0.456,  0.406 ], [0.229,  0.224,  0.225 ]),
    "synthetic": ([0.0,    0.0,    0.0   ], [1.0,    1.0,    1.0   ]),  # 合成数据已是标准正态分布
}

_NUM_CLASSES: dict[str, int] = {
    "cifar10": 10, "cifar100": 100, "imagenet": 1000, "synthetic": 10,
}

_DATASET_SUBDIR: dict[str, str] = {
    "cifar10": "CIFAR10", "cifar100": "CIFAR100",
}


# ── 内部辅助函数 ──────────────────────────────────────────────────────────────
def _build_transform(dataset_name: str, image_size: int, train: bool):
    """根据数据集名称和数据划分，返回对应的预处理流水线。"""
    if not _HAS_TORCHVISION:
        return None

    mean, std = _NORM_STATS.get(dataset_name.lower(), _NORM_STATS["imagenet"])

    if train:
        # 训练阶段加入随机数据增强，提升泛化性
        return tv_transforms.Compose([
            tv_transforms.Resize((image_size, image_size)),
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean, std),
        ])
    # 验证阶段只做确定性变换，保证结果可复现
    return tv_transforms.Compose([
        tv_transforms.Resize((image_size, image_size)),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean, std),
    ])


def _load_real_dataset(
    name: str, root: Path, train: bool, image_size: int
) -> Dataset | None:
    """加载真实数据集。如果路径不存在则返回 None。"""
    if not _HAS_TORCHVISION:
        return None

    transform = _build_transform(name, image_size, train)
    key = name.lower()

    if key in _DATASET_SUBDIR:
        dataset_root = root / _DATASET_SUBDIR[key]
        if not dataset_root.exists():
            return None
        cls = tv_datasets.CIFAR10 if key == "cifar10" else tv_datasets.CIFAR100
        return cls(dataset_root, train=train, download=False, transform=transform)

    if key == "imagenet":
        split_dir = root / "ImageNet" / ("train" if train else "val")
        if not split_dir.exists():
            return None
        return tv_datasets.ImageFolder(split_dir, transform=transform)

    return None


from collections.abc import Sized
def _maybe_subset(dataset: Dataset, limit: int | None) -> Dataset:
    """如有需要，从头截取 limit 个样本。返回类型仍满足 Dataset。"""
    if limit is None or limit <= 0:
        return dataset
    if not isinstance(dataset, Sized):
        return dataset
    if len(dataset) <= limit:
        return dataset
    # Subset 实现了 __len__，因此同样满足 Dataset 协议
    return Subset(dataset, list(range(limit)))


# ── 合成数据集 ────────────────────────────────────────────────────────────────
class SyntheticDataset(Dataset):
    """无需真实数据即可测试完整流水线的假数据集。
    对于相同的 index，始终返回相同的图像和标签，从而保证可复现性。
    """

    def __init__(self, size: int, num_classes: int, image_size: int, seed: int = 42) -> None:
        self.size        = size
        self.num_classes = num_classes
        self.image_size  = image_size
        self.seed        = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        g = torch.Generator().manual_seed(self.seed + index)
        image = torch.randn(3, self.image_size, self.image_size, generator=g)
        label = int(torch.randint(0, self.num_classes, (1,), generator=g).item())
        return image, label


# ── 对外公开的 API ─────────────────────────────────────────────────────────────
def make_dataloaders(
    dataset_name: str,
    data_root: str | Path,
    batch_size: int,
    image_size: int,
    num_classes:          int | None = None,
    num_workers:          int = 0,
    synthetic_if_missing: bool = True,
    train_size:           int | None = None,
    val_size:             int | None = None,
    seed:                 int = 42,
) -> DatasetBundle:
    """创建一对 DataLoader 并以 DatasetBundle 的形式返回。

    若数据集不存在：
    - synthetic_if_missing=True 时，自动退回到合成数据；
    - synthetic_if_missing=False 时，抛出 FileNotFoundError。
    """
    root = Path(data_root)
    num_classes = num_classes or _NUM_CLASSES.get(dataset_name.lower(), 10)

    train_ds = _load_real_dataset(dataset_name, root, train=True,  image_size=image_size)
    val_ds   = _load_real_dataset(dataset_name, root, train=False, image_size=image_size)
    source   = dataset_name

    # 数据集加载失败时的降级处理
    if train_ds is None or val_ds is None:
        if not synthetic_if_missing:
            raise FileNotFoundError(
                f"在 '{root}' 下未找到数据集 '{dataset_name}'。"
                "若要使用合成数据，请将 synthetic_if_missing 设为 True。"
            )
        source   = "synthetic"
        train_ds = SyntheticDataset(train_size or 512, num_classes, image_size, seed=seed)
        val_ds   = SyntheticDataset(val_size   or 256, num_classes, image_size, seed=seed + 10_000)

    # 按需截断数据集大小（用于快速调试）
    train_ds = _maybe_subset(train_ds, train_size)
    val_ds   = _maybe_subset(val_ds,   val_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_ds, 
        val_dataset=val_ds,      
        num_classes=num_classes,
        input_size=image_size,
        source=source,
    )



if __name__ == "__main__":
    import yaml

    # 从 yaml 读取配置，再传给 make_dataloaders——这是实验脚本的标准调用模式
    cfg_path = Path("_configs/default_env.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    print(f"配置文件加载自：{cfg_path}")
    print(f"  data_root  : {cfg['paths']['data_root']}")
    print(f"  dataset    : {cfg['data']['default_dataset']}")
    print(f"  image_size : {cfg['data']['default_image_size']}")

    bundle = make_dataloaders(
        dataset_name         = cfg["data"]["default_dataset"],
        data_root            = cfg["paths"]["data_root"],
        batch_size           = 32,
        image_size           = cfg["data"]["default_image_size"],
        num_workers          = cfg["num_workers"],
        synthetic_if_missing = cfg["data"]["synthetic_if_missing"],
        train_size           = cfg["data"]["synthetic_train_size"],
        val_size             = cfg["data"]["synthetic_val_size"],
        seed                 = cfg["seed"],
    )

    print(f"\nDatasetBundle 创建成功：")
    print(f"  source       : {bundle.source}")
    print(f"  num_classes  : {bundle.num_classes}")
    print(f"  input_size   : {bundle.input_size}")
    print(f"  训练批次数    : {len(bundle.train_loader)}")
    print(f"  验证批次数    : {len(bundle.val_loader)}")

    # 取第一个 batch，验证整条流水线的输出形状是否符合预期
    images, labels = next(iter(bundle.train_loader))
    print(f"\n第一个 batch：")
    print(f"  images shape : {images.shape}")          # 期望 (B, 3, H, W)
    print(f"  labels shape : {labels.shape}")          # 期望 (B,)
    print(f"  image dtype  : {images.dtype}")          # 期望 torch.float32
    print(f"  label 范围   : [{labels.min()}, {labels.max()}]")

    print("\n✓ datasets.py 冒烟测试通过。")