""" Src/Collab_System/tensor_transfer.py
    中间激活的序列化 / 反序列化工具。
    模拟边缘端将激活打包成字节流传给云端的过程，用于估算实际传输的 payload 大小。
"""
import io
import zlib
import torch


def serialize_tensor(
    tensor: torch.Tensor,
    compress: bool = False,
    method: str = "zlib",
) -> bytes:
    """
    将张量序列化为字节流，可选 zlib 压缩。
    先用 torch.save 把张量写入内存 buffer（含元数据：dtype、shape、stride），再按需压缩，返回最终要"传输"的 payload。
    """
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)   # detach 防止携带梯度信息
    payload = buffer.getvalue()
    if compress and method == "zlib":
        payload = zlib.compress(payload)
    return payload


def deserialize_tensor(
    payload: bytes,
    compress: bool = False,
    method: str = "zlib",
) -> torch.Tensor:
    """
    将字节流还原为张量，与 serialize_tensor 对称
    """
    if compress and method == "zlib":
        payload = zlib.decompress(payload)
    return torch.load(io.BytesIO(payload), map_location="cpu")