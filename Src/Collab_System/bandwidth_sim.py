"""Src/Collab_System/bandwidth_sim.py"""


def estimate_transfer_time_ms(
    data_bytes: int,
    bandwidth_mbps: float,
    protocol_overhead_ms: float = 2.0,
) -> float:
    """网络传输延迟
    - 输入：传输字节数
    - 输出：传输延迟
    - transfer_time_ms = (bytes / bandwidth_bytes_per_ms) + protocol_overhead_ms
        - bandwidth_bytes_per_ms: bandwidth_mbps * 1e6 / 8 / 1000
        - protocol_overhead_ms: 协议固定开销，默认 2.0 ms，对应轻量级 gRPC 场景
    """
    # Mbps → bytes/ms：1 Mbps = 10^6 bits/s = 10^6 / 8 bytes/s = 125 bytes/ms
    bandwidth_bytes_per_ms = bandwidth_mbps * 125.0
    transfer_ms = data_bytes / bandwidth_bytes_per_ms
    return transfer_ms + protocol_overhead_ms


def saved_transfer_ratio(dag_bytes: int, chain_bytes: int) -> float:
    """链式拓扑相对于 DAG 拓扑节省的传输量比例"""
    if dag_bytes <= 0:
        return 0.0
    return (dag_bytes - chain_bytes) / dag_bytes
