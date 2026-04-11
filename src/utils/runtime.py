from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ensure_dir, resolve_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output_path = resolve_path(path)
    ensure_dir(output_path.parent)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.element_size() * tensor.numel()


def safe_item(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)
