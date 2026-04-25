""" Src/DatasetsProcess/datasets.py """
import json
import random
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets as tv_datasets
from torchvision import transforms as tv_transforms


# ══════════════════════════════════════════════════════════════════════════════
# § 1  全局配置
# ══════════════════════════════════════════════════════════════════════════════

# 各数据集的 ImageNet 风格归一化参数（mean / std，RGB 三通道）
_NORM_STATS: dict[str, tuple[list[float], list[float]]] = {
    "cifar10":     ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    "cifar100":    ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    "imagenet":    ([0.485,  0.456,  0.406 ], [0.229,  0.224,  0.225 ]),
    "imagenet100": ([0.485,  0.456,  0.406 ], [0.229,  0.224,  0.225 ]),
}

# 各数据集默认类别数，make_dataloaders 未显式指定时使用
_NUM_CLASSES: dict[str, int] = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet": 1000,
    "imagenet100": 100,
}

@dataclass
class DatasetBundle:
    """make_dataloaders 的返回值，实验脚本直接解包使用。"""
    train_loader:  DataLoader
    val_loader:    DataLoader
    train_dataset: Dataset
    val_dataset:   Dataset
    num_classes:   int
    input_size:    int
    source:        str          # 数据集名称（小写），如 "cifar10"


# ══════════════════════════════════════════════════════════════════════════════
# § 2  数据预处理
# ══════════════════════════════════════════════════════════════════════════════

def _build_transform(dataset_name: str, image_size: int, train: bool):
    """构造对应数据集与训练/评估阶段的图像预处理流水线。

    - 训练集：Resize → RandomHorizontalFlip → ToTensor → Normalize
    - 验证集：Resize → ToTensor → Normalize（去除随机增强）
    """
    mean, std = _NORM_STATS.get(dataset_name.lower(), _NORM_STATS["imagenet"])
    if train:
        return tv_transforms.Compose([
            tv_transforms.Resize((image_size, image_size)),
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean, std),
        ])
    return tv_transforms.Compose([
        tv_transforms.Resize((image_size, image_size)),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean, std),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# § 3  CIFAR 数据集加载
# ══════════════════════════════════════════════════════════════════════════════

def _load_cifar(
    name: str, data_root: Path, train: bool, image_size: int
) -> Dataset | None:
    """加载 CIFAR-10 / CIFAR-100。

    期望目录结构：
        data_root/
          CIFAR10/   ← cifar10
          CIFAR100/  ← cifar100

    路径不存在时返回 None，由上层决定是否抛错。
    """
    subdir = {"cifar10": "CIFAR10", "cifar100": "CIFAR100"}[name]
    root = data_root / subdir
    if not root.exists():
        return None
    cls = tv_datasets.CIFAR10 if name == "cifar10" else tv_datasets.CIFAR100
    return cls(
        str(root),
        train=train,
        download=False,
        transform=_build_transform(name, image_size, train=train),
    )


# ══════════════════════════════════════════════════════════════════════════════
# § 4  ImageNet100/1000 数据集加载
# ══════════════════════════════════════════════════════════════════════════════

def _imagenet100_paths(data_root: Path) -> tuple[Path, Path]:
    """返回 ImageNet100 的 (图片根目录, 切分索引文件) 路径。"""
    return (
        data_root / "ImageNet100" / "imagenet100",
        data_root / "ImageNet100" / "split_indices.json",
    )


def _load_split(data_root: Path) -> dict:
    """从磁盘读取 ImageNet100 切分索引 JSON 文件。"""
    _, split_file = _imagenet100_paths(data_root)
    with open(split_file) as f:
        return json.load(f)


def generate_imagenet100_split(
    data_root: str | Path,
    val_ratio: float = 0.2,
    seed:      int   = 42,
) -> None:
    """对 ImageNet100 做一次性 train/val 随机切分，结果以索引列表形式"""
    img_root, split_file = _imagenet100_paths(Path(data_root))
    dataset = tv_datasets.ImageFolder(root=str(img_root), transform=tv_transforms.ToTensor())
    total = len(dataset)
    val_size   = int(val_ratio * total)
    train_size = total - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w") as f:
        json.dump({
            "seed":          seed,
            "val_ratio":     val_ratio,
            "total":         total,
            "train_size":    train_size,
            "val_size":      val_size,
            "train_indices": train_subset.indices,
            "val_indices":   val_subset.indices,
        }, f)
    print(f"索引已保存：{split_file}")


def _load_imagenet100(
    data_root: Path, train: bool, image_size: int
) -> Dataset | None:
    """加载 ImageNet100 的一个 split，图片目录或索引文件不存在时返回 None。"""
    img_root, split_file = _imagenet100_paths(data_root)
    if not img_root.exists() or not split_file.exists():
        return None

    split = _load_split(data_root)
    transform = _build_transform("imagenet100", image_size, train=train)
    base = tv_datasets.ImageFolder(str(img_root), transform=transform)
    indices = split["train_indices"] if train else split["val_indices"]
    return Subset(base, indices)


def _load_imagenet1000(
    data_root: Path, train: bool, image_size: int
) -> Dataset | None:
    """加载完整 ImageNet（ILSVRC2012），目录按官方格式解压
        data_root/ImageNet/train/  
        data_root/ImageNet/val/    
    """
    split_dir = data_root / "ImageNet" / ("train" if train else "val")
    if not split_dir.exists():
        return None
    return tv_datasets.ImageFolder(
        str(split_dir),
        transform=_build_transform("imagenet", image_size, train=train),
    )


# ══════════════════════════════════════════════════════════════════════════════
# § 5  统一数据集调用入口
# ══════════════════════════════════════════════════════════════════════════════

def _load_dataset_pair(
    dataset_name: str,
    data_root: Path,
    image_size: int,
) -> tuple[Dataset | None, Dataset | None]:
    """按数据集名称路由到对应的 (train, val) 加载函数"""
    if dataset_name == "imagenet100":
        return (
            _load_imagenet100(data_root, train=True,  image_size=image_size),
            _load_imagenet100(data_root, train=False, image_size=image_size),
        )
    if dataset_name == "imagenet":          # ← 新增
        return (
            _load_imagenet1000(data_root, train=True,  image_size=image_size),
            _load_imagenet1000(data_root, train=False, image_size=image_size),
        )
    if dataset_name in ("cifar10", "cifar100"):
        return (
            _load_cifar(dataset_name, data_root, train=True,  image_size=image_size),
            _load_cifar(dataset_name, data_root, train=False, image_size=image_size),
        )
    return None, None


def make_dataloaders(
    dataset_name: str,
    data_root:    str | Path,
    batch_size:   int,
    image_size:   int,
    num_classes:  int | None = None,
    num_workers:  int        = 0,
    seed:         int        = 42,
) -> DatasetBundle:
    """创建一对 DataLoader，以 DatasetBundle 形式返回

    支持的数据集：
        - cifar10 / cifar100：直接读已下载的目录（download=False）。
        - imagenet100：依赖 generate_split() 生成的索引文件进行切分。

    Args:
        dataset_name: 数据集名称（大小写不敏感），如 "cifar10"、"imagenet100"。
        data_root:    数据集根目录，各子数据集以子目录形式存放。
        batch_size:   DataLoader 的 batch size。
        image_size:   输入图像边长（正方形 resize）。
        num_classes:  类别数，为 None 时从 _NUM_CLASSES 查表。
        num_workers:  DataLoader 的工作进程数。
        seed:         保留字段，供将来需要随机性的操作使用。

    Raises:
        FileNotFoundError: 数据集目录或切分索引文件不存在时抛出
    """
    root        = Path(data_root)
    key         = dataset_name.lower()
    num_classes = num_classes or _NUM_CLASSES.get(key, 10)

    train_ds, val_ds = _load_dataset_pair(key, root, image_size)
    if train_ds is None or val_ds is None:
        raise FileNotFoundError(
            f"在 '{root}' 下未找到数据集 '{dataset_name}'。\n"
            "若使用 imagenet100，请先运行：\n"
            "  from Src.DatasetsProcess.datasets import generate_imagenet100_split\n"
            "  generate_imagenet100_split(data_root)"
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_classes=num_classes,
        input_size=image_size,
        source=key,
    )


# ══════════════════════════════════════════════════════════════════════════════
# § 6  校准集构建（用于 PTQ / 补偿器训练）
# ══════════════════════════════════════════════════════════════════════════════

def sample_calibration_subset(dataset: Dataset, calib_size: int, seed: int = 42) -> Dataset:
    """从完整数据集中随机采样 calib_size 个样本，返回 Subset。

    Args:
        dataset:    源数据集，必须实现 __len__。
        calib_size: 采样数量；若 ≥ len(dataset) 则返回原始数据集。
        seed:       随机种子，保证采样可复现。
    """
    if not isinstance(dataset, Sized):
        raise TypeError("dataset 必须实现 __len__")
    if calib_size <= 0 or calib_size >= len(dataset):
        return dataset
    indices = random.Random(seed).sample(range(len(dataset)), k=calib_size)
    return Subset(dataset, indices)


def build_calibration_loader(
    dataset:     Dataset,
    calib_size:  int,
    batch_size:  int,
    num_workers: int = 0,
    seed:        int = 42,
) -> DataLoader:
    """构造校准集 DataLoader，供 PTQ 量化或补偿器 MSE 预热训练使用。

    内部调用 sample_calibration_subset 完成采样，shuffle=True 以保证
    每个 batch 的类别分布尽量均匀。
    """
    subset = sample_calibration_subset(dataset, calib_size=calib_size, seed=seed)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)