from __future__ import annotations

import io
import zlib

import torch


def serialize_tensor(
    tensor: torch.Tensor,
    compress: bool = False,
    method: str = "zlib",
) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    payload = buffer.getvalue()
    if compress and method == "zlib":
        payload = zlib.compress(payload)
    return payload


def deserialize_tensor(payload: bytes, compress: bool = False, method: str = "zlib") -> torch.Tensor:
    if compress and method == "zlib":
        payload = zlib.decompress(payload)
    return torch.load(io.BytesIO(payload), map_location="cpu")
