"""src/system/bandwidth_sim.py — 网络传输延迟仿真。

这个模块把"传输多少字节、带宽是多少"翻译成"需要等多少毫秒"，
是 Exp1 系统开销分析和 Exp3 端到端延迟分解的共同基础。

公式：transfer_time_ms = (bytes / bandwidth_bytes_per_ms) + protocol_overhead_ms

其中 bandwidth_bytes_per_ms = bandwidth_mbps * 1e6 / 8 / 1000
"""
from __future__ import annotations


def estimate_transfer_time_ms(
    data_bytes: int,
    bandwidth_mbps: float,
    protocol_overhead_ms: float = 2.0,
) -> float:
    """根据数据量和带宽，估算网络传输延迟（毫秒）。

    Args:
        data_bytes:           需要传输的数据字节数。
        bandwidth_mbps:       信道带宽，单位 Mbps（兆比特/秒）。
        protocol_overhead_ms: 协议固定开销（握手、帧头等），单位毫秒。
                              默认 2.0 ms，对应轻量级 gRPC 场景。

    Returns:
        估算的总传输延迟，单位毫秒。
    """
    if bandwidth_mbps <= 0:
        raise ValueError(f"bandwidth_mbps 必须为正数，实际值：{bandwidth_mbps}")

    # Mbps → bytes/ms：1 Mbps = 10^6 bits/s = 10^6 / 8 bytes/s = 125 bytes/ms
    bandwidth_bytes_per_ms = bandwidth_mbps * 125.0
    transfer_ms = data_bytes / bandwidth_bytes_per_ms
    return transfer_ms + protocol_overhead_ms


def saved_transfer_ratio(dag_bytes: int, chain_bytes: int) -> float:
    """计算链式拓扑相对于 DAG 拓扑节省的传输量比例。

    返回值 > 0 表示链式更优；= 0.5 表示节省了一半（典型残差块场景）。
    """
    if dag_bytes <= 0:
        return 0.0
    return (dag_bytes - chain_bytes) / dag_bytes