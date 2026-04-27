# Src/Metrics/max_batch.py

from typing import Any

import torch
from torch import nn


def find_max_batch_size(
    model: nn.Module,
    sample_single: torch.Tensor,  # shape [1, C, H, W]，单张图
    min_bs: int = 1,
    max_bs: int = 1024,
    **forward_kwargs: Any,
) -> int:
    """
    二分查找：在当前显存下该配置最大能跑多大的 batch
    返回最大可用 batch size，OOM 则返回 0
    """
    model.eval()

    def can_run(bs: int) -> bool:
        torch.cuda.empty_cache()
        batch = sample_single.repeat(bs, 1, 1, 1)  # 真实分配 bs 份内存
        try:
            with torch.no_grad():
                model(batch, **forward_kwargs)
            return True
        except torch.cuda.OutOfMemoryError:
            return False

    # 先确认 min_bs 可行
    if not can_run(min_bs):
        return 0

    # 二分查找
    lo, hi = min_bs, max_bs
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if can_run(mid):
            lo = mid
        else:
            hi = mid - 1

    return lo
