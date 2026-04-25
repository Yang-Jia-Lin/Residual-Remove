""" Src/DatasetsProcess/datasets.py
    数据集加载与 DataLoader 创建
    目录结构（data_root）：
        ImageNet100/
            imagenet100/          ← ImageFolder 根目录（每个类一个子文件夹）
            split_indices.json    ← generate_split() 生成（首次使用前需运行一次）
        CIFAR10/
        CIFAR100/
"""
import json
import random
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from torchvision import datasets as tv_datasets
from torchvision import transforms as tv_transforms



# ── 归一化统计量 ──────────────────────────────────────────────────────────────
_NORM_STATS: dict[str, tuple[list[float], list[float]]] = {
    "cifar10":      ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    "cifar100":     ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    "imagenet":     ([0.485,  0.456,  0.406 ], [0.229,  0.224,  0.225 ]),
    "imagenet100":  ([0.485,  0.456,  0.406 ], [0.229,  0.224,  0.225 ]),
    "synthetic":    ([0.0,    0.0,    0.0   ], [1.0,    1.0,    1.0   ]),
}

_NUM_CLASSES: dict[str, int] = {
    "cifar10": 10, 
    "cifar100": 100,
    "imagenet": 1000, 
    "imagenet100": 100,
    "synthetic": 10,
}

@dataclass
class DatasetBundle:
    """make_dataloaders() 的返回值，供实验脚本直接解包使用。"""
    train_loader:  DataLoader
    val_loader:    DataLoader
    train_dataset: Dataset
    val_dataset:   Dataset
    num_classes:   int
    input_size:    int
    source:        str


# ── 合成数据集 ────────────────────────────────────────────────────────────────
class SyntheticDataset(Dataset):
    """无需真实数据即可跑通完整流水线的占位数据集。
    相同 index 始终返回相同的图像和标签，保证可复现。
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


# ── 内部辅助 ──────────────────────────────────────────────────────────────────
def _build_transform(dataset_name: str, image_size: int, train: bool):
    """返回对应数据集和阶段的预处理流水线。"""
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


def _maybe_subset(dataset: Dataset, limit: int | None) -> Dataset:
    """从头截取 limit 个样本，用于快速调试。"""
    if limit is None or limit <= 0:
        return dataset
    if not isinstance(dataset, Sized):
        return dataset
    if len(dataset) <= limit:
        return dataset
    return Subset(dataset, list(range(limit)))


# ── ImageNet100 索引切分 ──────────────────────────────────────────────────────
def _imagenet100_paths(data_root: Path) -> tuple[Path, Path]:
    """返回 (imagenet100 图片根目录, split 索引文件路径)。"""
    return (
        data_root / "ImageNet100" / "imagenet100",
        data_root / "ImageNet100" / "split_indices.json",
    )


def generate_split(
    data_root:  str | Path,
    val_ratio:  float = 0.2,
    seed:       int   = 42,
) -> None:
    """对 ImageNet100 做一次性 train/val 切分，结果保存为索引文件。首次使用前必须运行一次"""
    img_root, split_file = _imagenet100_paths(Path(data_root))
    
    print(f"[1/3] 扫描目录：{img_root}")
    dataset = tv_datasets.ImageFolder(root=str(img_root), transform=tv_transforms.ToTensor())
    total, n_cls = len(dataset), len(dataset.classes)
    print(f"      共 {total} 张图，{n_cls} 个类别")
    
    val_size   = int(val_ratio * total)
    train_size = total - val_size
    print(f"[2/3] 按 {int((1-val_ratio)*100)}/{int(val_ratio*100)} 切分，seed={seed}")
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
    print(f"[3/3] 索引已保存：{split_file}")
    print(f"\n✅ 完成：train={train_size}，val={val_size}")


def verify_split(data_root: str | Path) -> None:
    """验证已有 split 索引文件，检查 train/val 无交集。"""
    _, split_file = _imagenet100_paths(Path(data_root))
    print(f"读取索引：{split_file}")
    with open(split_file) as f:
        split = json.load(f)

    print(f"  seed={split['seed']}  total={split['total']}")
    print(f"  train={split['train_size']}  val={split['val_size']}")

    overlap = set(split["train_indices"]) & set(split["val_indices"])
    if overlap:
        print(f"\n❌ 存在 {len(overlap)} 个重叠索引，请重新生成！")
    else:
        print(f"\n✅ 切分干净，无数据泄漏")


def get_imagenet100_datasets(
    data_root:       str | Path,
    image_size:      int = 224,
    train_transform=None,
    val_transform=None,
) -> tuple[Dataset, Dataset]:
    """直接返回固定切分的 ImageNet100 train/val Dataset，供外部 import 使用。"""
    img_root, split_file = _imagenet100_paths(Path(data_root))
    if not split_file.exists():
        raise FileNotFoundError(
            f"未找到切分索引：{split_file}\n"
            "请先运行 generate_split(data_root) 生成索引文件。"
        )

    with open(split_file) as f:
        split = json.load(f)
    # train/val 各自加载一份，transform 互不干扰
    train_ds = tv_datasets.ImageFolder(
        str(img_root),
        transform=train_transform or _build_transform("imagenet100", image_size, train=True),
    )
    val_ds = tv_datasets.ImageFolder(
        str(img_root),
        transform=val_transform or _build_transform("imagenet100", image_size, train=False),
    )
    return (
        Subset(train_ds, split["train_indices"]),
        Subset(val_ds,   split["val_indices"]),
    )


def _load_imagenet100(
    data_root: Path, train: bool, image_size: int
) -> Dataset | None:
    """内部调用：加载 ImageNet100 的一个 split，找不到索引文件则返回 None。"""
    img_root, split_file = _imagenet100_paths(data_root)
    if not img_root.exists() or not split_file.exists():
        return None

    with open(split_file) as f:
        split = json.load(f)

    transform = _build_transform("imagenet100", image_size, train=train)
    base      = tv_datasets.ImageFolder(str(img_root), transform=transform)
    indices   = split["train_indices"] if train else split["val_indices"]
    return Subset(base, indices)


def _load_cifar(
    name: str, data_root: Path, train: bool, image_size: int
) -> Dataset | None:
    """内部调用：加载 CIFAR10/100，路径不存在则返回 None。"""
    subdir = {"cifar10": "CIFAR10", "cifar100": "CIFAR100"}[name]
    root   = data_root / subdir
    if not root.exists():
        return None
    cls = tv_datasets.CIFAR10 if name == "cifar10" else tv_datasets.CIFAR100
    return cls(str(root), train=train, download=False,
               transform=_build_transform(name, image_size, train=train))


# ── 主 API ──────────────────────────────────────────────────────────
def make_dataloaders(
    dataset_name: str,
    data_root:    str | Path,
    batch_size:   int,
    image_size:   int,
    num_classes:          int | None = None,
    num_workers:          int        = 0,
    synthetic_if_missing: bool       = True,
    train_size:           int | None = None,
    val_size:             int | None = None,
    seed:                 int        = 42,
) -> DatasetBundle:
    """ 创建一对 DataLoader，以 DatasetBundle 形式返回。
        ImageNet100 使用索引文件切分（需提前运行 generate_split）。
        CIFAR10/100 直接读已下载的数据集目录。
        数据集缺失时：synthetic_if_missing=True 退回合成数据，否则抛 FileNotFoundError。
    """
    root        = Path(data_root)
    key         = dataset_name.lower()
    num_classes = num_classes or _NUM_CLASSES.get(key, 10)

    # 按数据集名称分发加载逻辑
    if key == "imagenet100":
        train_ds = _load_imagenet100(root, train=True,  image_size=image_size)
        val_ds   = _load_imagenet100(root, train=False, image_size=image_size)
    elif key in ("cifar10", "cifar100"):
        train_ds = _load_cifar(key, root, train=True,  image_size=image_size)
        val_ds   = _load_cifar(key, root, train=False, image_size=image_size)
    else:
        train_ds = val_ds = None

    source = key

    # 数据集缺失时的降级处理
    if train_ds is None or val_ds is None:
        if not synthetic_if_missing:
            raise FileNotFoundError(
                f"在 '{root}' 下未找到数据集 '{dataset_name}'。\n"
                "若使用 imagenet100，请确认已运行 generate_split() 并检查目录结构。"
            )
        source   = "synthetic"
        train_ds = SyntheticDataset(train_size or 512, num_classes, image_size, seed=seed)
        val_ds   = SyntheticDataset(val_size   or 256, num_classes, image_size, seed=seed + 10_000)

    # 按需截断（快速调试）
    train_ds = _maybe_subset(train_ds, train_size)
    val_ds   = _maybe_subset(val_ds,   val_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_classes=num_classes,
        input_size=image_size,
        source=source,
    )


def sample_calibration_subset(dataset: Dataset, calib_size: int, seed: int = 42) -> Dataset:
    if not isinstance(dataset, Sized):
        raise TypeError("dataset must implement __len__")
    if calib_size <= 0 or calib_size >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return Subset(dataset, indices[:calib_size])


def build_calibration_loader(
    dataset: Dataset,
    calib_size: int,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    subset = sample_calibration_subset(dataset, calib_size=calib_size, seed=seed)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
