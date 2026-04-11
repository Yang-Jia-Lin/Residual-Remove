from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from torchvision import datasets as tv_datasets
    from torchvision import transforms as tv_transforms
except Exception:  # pragma: no cover
    tv_datasets = None
    tv_transforms = None


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    train_dataset: Dataset
    val_dataset: Dataset
    num_classes: int
    input_size: int
    source: str


class SyntheticClassificationDataset(Dataset):
    def __init__(self, size: int, num_classes: int, image_size: int, seed: int = 42) -> None:
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.randn(3, self.image_size, self.image_size, generator=generator)
        label = int(torch.randint(0, self.num_classes, (1,), generator=generator).item())
        return image, label


def _default_num_classes(dataset_name: str) -> int:
    return {"cifar10": 10, "cifar100": 100, "imagenet": 1000, "synthetic": 10}.get(dataset_name.lower(), 10)


def _build_transform(image_size: int):
    if tv_transforms is None:
        return None
    return tv_transforms.Compose([tv_transforms.Resize((image_size, image_size)), tv_transforms.ToTensor()])


def _load_dataset(name: str, root: Path, train: bool, image_size: int) -> Dataset | None:
    if tv_datasets is None:
        return None
    transform = _build_transform(image_size)
    key = name.lower()
    if key == "cifar10":
        dataset_root = root / "CIFAR10"
        if dataset_root.exists():
            return tv_datasets.CIFAR10(dataset_root, train=train, download=False, transform=transform)
    if key == "cifar100":
        dataset_root = root / "CIFAR100"
        if dataset_root.exists():
            return tv_datasets.CIFAR100(dataset_root, train=train, download=False, transform=transform)
    if key == "imagenet":
        dataset_root = root / "ImageNet" / ("train" if train else "val")
        if dataset_root.exists():
            return tv_datasets.ImageFolder(dataset_root, transform=transform)
    return None


def _maybe_subset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit <= 0 or len(dataset) <= limit:
        return dataset
    return Subset(dataset, list(range(limit)))


def make_dataloaders(
    dataset_name: str,
    data_root: str | Path,
    batch_size: int,
    image_size: int,
    num_classes: int | None = None,
    num_workers: int = 0,
    synthetic_if_missing: bool = True,
    train_size: int = 512,
    val_size: int = 256,
    seed: int = 42,
) -> DatasetBundle:
    root = Path(data_root)
    num_classes = num_classes or _default_num_classes(dataset_name)

    train_dataset = _load_dataset(dataset_name, root, train=True, image_size=image_size)
    val_dataset = _load_dataset(dataset_name, root, train=False, image_size=image_size)
    source = dataset_name

    if train_dataset is None or val_dataset is None:
        if not synthetic_if_missing:
            raise FileNotFoundError(f"Dataset '{dataset_name}' is unavailable under {root}.")
        source = "synthetic"
        train_dataset = SyntheticClassificationDataset(train_size, num_classes, image_size, seed=seed)
        val_dataset = SyntheticClassificationDataset(val_size, num_classes, image_size, seed=seed + 10_000)

    train_dataset = _maybe_subset(train_dataset, train_size)
    val_dataset = _maybe_subset(val_dataset, val_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_classes=num_classes,
        input_size=image_size,
        source=source,
    )
