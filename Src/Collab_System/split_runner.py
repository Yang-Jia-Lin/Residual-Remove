"""Src/Collab_System/split_runner.py
端边云协同推理时延仿真
端计算 → 传输 → 边缘计算 → 传输 → 云端计算
"""

import time
from typing import Any

import torch

from Src.Collab_System.bandwidth_sim import estimate_transfer_time_ms
from Src.Collab_System.tensor_transfer import serialize_tensor


def _timed_call(fn, device: torch.device) -> tuple[Any, float]:
    """
    计算一次函数调用的耗时（毫秒）
        CUDA: 计时前后各 synchronize，否则 CUDA 异步执行会导致计时偏小
        CPU: 下直接用 perf_counter
    """
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
    split_point: str,  # 分割点名称，对应模型中的层/block 标识
    bandwidth_mbps: float,  # 模拟的传输带宽（Mbps）
    protocol_overhead_ms: float = 0.0,  # 协议固定开销（握手、头部等），默认不加
    compress: bool = False,  # 是否对中间激活做压缩
    compression_method: str = "zlib",  # 压缩算法，compress=True 时生效
    mode: str = "full",  # 推理模式："full"=完整网络，"removed"=删除残差后
    removed_blocks: list[str]
    | None = None,  # removed 模式下指定哪些 block 的残差连接被删除
) -> dict[str, float]:
    """
    模拟一次完整的分割推理流程，返回各阶段时延。

    流程：
        1. edge 段：model.forward_to_split()  —— 边缘设备执行 split_point 之前的层
        2. transfer：将中间激活序列化，按带宽估算传输时延
        3. cloud 段：model.forward_from_split() —— 云端执行 split_point 之后的层

    Args:
        model:        已实现 forward_to_split / forward_from_split 接口的模型
        sample:       单条输入张量（含 batch 维）
        split_point:  分割点标识，由模型内部解析
        bandwidth_mbps:        链路带宽
        protocol_overhead_ms:  固定协议开销
        compress:              是否压缩中间激活
        compression_method:    压缩方式
        mode:                  "full" 或 "removed"
        removed_blocks:        mode="removed" 时需删除残差的 block 列表

    Returns:
        edge_ms      —— 边缘推理耗时
        transfer_ms  —— 传输耗时（含压缩后 payload 大小估算）
        cloud_ms     —— 云端推理耗时
        total_ms     —— 三者之和
        payload_bytes —— 序列化后的激活大小（字节）
    """
    # 模型必须实现分割推理接口
    if not hasattr(model, "forward_to_split") or not hasattr(
        model, "forward_from_split"
    ):
        raise AttributeError(
            "Model must expose forward_to_split() and forward_from_split()."
        )

    model.eval()
    device = sample.device

    with torch.no_grad():
        # ── Stage 1: 边缘端推理（split_point 之前的层）──────────────────────
        activation, edge_ms = _timed_call(
            lambda: model.forward_to_split(
                sample,
                split_point=split_point,
                mode=mode,
                removed_blocks=removed_blocks,
            ),
            device,
        )

        # ── Stage 2: 激活序列化 + 带宽估算 ──────────────────────────────────
        # serialize_tensor 将 activation 转成字节串（可选压缩），模拟实际传输的 payload
        payload = serialize_tensor(
            activation, compress=compress, method=compression_method
        )
        transfer_ms = estimate_transfer_time_ms(
            len(payload),
            bandwidth_mbps=bandwidth_mbps,
            protocol_overhead_ms=protocol_overhead_ms,
        )

        # ── Stage 3: 云端推理（split_point 之后的层）────────────────────────
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
