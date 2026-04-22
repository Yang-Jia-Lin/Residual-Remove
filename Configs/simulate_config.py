"""Configs/simulate_config.py"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class SimulateConfig:
    bandwidth_mbps:       List[int] = field(default_factory=lambda: [1, 10, 100])
    protocol_overhead_ms: float     = 2.0
    edge_memory_limit_mb: int       = 4096
    edge_compute_factor:  float     = 8.0

# 全局实例
simulate_config = SimulateConfig()