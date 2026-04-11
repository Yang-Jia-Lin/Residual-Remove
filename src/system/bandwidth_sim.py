from __future__ import annotations


def estimate_transfer_time_ms(
    num_bytes: int,
    bandwidth_mbps: float,
    protocol_overhead_ms: float = 0.0,
) -> float:
    if bandwidth_mbps <= 0:
        raise ValueError("bandwidth_mbps must be positive.")
    bits = num_bytes * 8.0
    transfer_seconds = bits / (bandwidth_mbps * 1_000_000.0)
    return transfer_seconds * 1000.0 + protocol_overhead_ms
