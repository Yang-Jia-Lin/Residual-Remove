from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATASET_NUM_CLASSES = {
    "cifar10": 10,
}


def build_dataloaders(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, int]:
    dataset_name = dataset_name.lower()
    if dataset_name != "cifar10":
        raise ValueError("Currently only cifar10 is implemented in this scaffold.")

    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=train_transform,
        download=True,
    )
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        transform=test_transform,
        download=True,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader, DATASET_NUM_CLASSES[dataset_name]
