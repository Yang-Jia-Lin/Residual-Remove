from __future__ import annotations

import random

from torch.utils.data import DataLoader, Dataset, Subset


def sample_calibration_subset(dataset: Dataset, calib_size: int, seed: int = 42) -> Dataset:
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
