from __future__ import annotations

import time
from typing import Any

import torch


def measure_latency(
    model: torch.nn.Module,
    sample: torch.Tensor,
    repetitions: int = 20,
    warmup: int = 5,
    **forward_kwargs: Any,
) -> float:
    model.eval()
    device = sample.device
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample, **forward_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repetitions):
            _ = model(sample, **forward_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end = time.perf_counter()
    return (end - start) * 1000.0 / repetitions
