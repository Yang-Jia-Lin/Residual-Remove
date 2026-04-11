from __future__ import annotations


def estimate_energy_joules(
    macs: int,
    dram_bytes: int,
    dram_pj_per_byte: float,
    mac_pj: float,
) -> dict[str, float]:
    dram_j = dram_bytes * dram_pj_per_byte * 1e-12
    mac_j = macs * mac_pj * 1e-12
    return {"dram_j": dram_j, "mac_j": mac_j, "total_j": dram_j + mac_j}
