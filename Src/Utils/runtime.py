"""Src/Utils/runtime.py"""

import csv
import random
from pathlib import Path

import numpy as np
import torch


def extract_logits(output: torch.Tensor | dict) -> torch.Tensor:
    """从模型输出中提取 logits 张量"""
    if isinstance(output, dict):
        return output["logits"]
    return output


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def write_csv(output: str | Path, rows: list[dict]) -> Path:
    """把一组字典写成 CSV 文件，自动创建父目录"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("")
        return path

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return path


def tensor_bytes(tensor) -> int:
    """计算一个 Tensor 占用的字节数。"""
    return tensor.numel() * tensor.element_size()


def safe_item(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)
