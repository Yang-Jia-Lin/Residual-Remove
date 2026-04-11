from __future__ import annotations

import time
from typing import Any

import torch

from src.system.bandwidth_sim import estimate_transfer_time_ms
from src.system.tensor_transfer import serialize_tensor


def _timed_call(fn, device: torch.device) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end = time.perf_counter()
    return result, (end - start) * 1000.0


def run_split_inference(
    model: torch.nn.Module,
    sample: torch.Tensor,
    split_point: str,
    bandwidth_mbps: float,
    protocol_overhead_ms: float = 0.0,
    compress: bool = False,
    compression_method: str = "zlib",
    mode: str = "full",
    removed_blocks: list[str] | None = None,
) -> dict[str, float]:
    if not hasattr(model, "forward_to_split") or not hasattr(model, "forward_from_split"):
        raise AttributeError("Model must expose forward_to_split() and forward_from_split().")

    model.eval()
    device = sample.device
    with torch.no_grad():
        activation, edge_ms = _timed_call(
            lambda: model.forward_to_split(
                sample,
                split_point=split_point,
                mode=mode,
                removed_blocks=removed_blocks,
            ),
            device,
        )
        payload = serialize_tensor(activation, compress=compress, method=compression_method)
        transfer_ms = estimate_transfer_time_ms(
            len(payload),
            bandwidth_mbps=bandwidth_mbps,
            protocol_overhead_ms=protocol_overhead_ms,
        )
        _, cloud_ms = _timed_call(
            lambda: model.forward_from_split(
                activation,
                split_point=split_point,
                mode=mode,
                removed_blocks=removed_blocks,
            ),
            device,
        )

    return {
        "edge_ms": edge_ms,
        "transfer_ms": transfer_ms,
        "cloud_ms": cloud_ms,
        "total_ms": edge_ms + transfer_ms + cloud_ms,
        "payload_bytes": float(len(payload)),
    }
