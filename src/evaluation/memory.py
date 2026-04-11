from __future__ import annotations

import torch

from src.evaluation.accuracy import extract_logits
from src.utils.runtime import tensor_bytes


def parameter_bytes(model: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def measure_peak_memory(
    model: torch.nn.Module,
    sample: torch.Tensor,
    **forward_kwargs,
) -> dict[str, float]:
    model.eval()
    device = sample.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(sample, **forward_kwargs)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        return {"peak_bytes": float(peak_bytes), "peak_mb": float(peak_bytes) / (1024**2)}

    with torch.no_grad():
        output = model(sample, return_features=True, **forward_kwargs)
    logits = extract_logits(output)
    features = output["features"] if isinstance(output, dict) else {}
    activation_peak = max((tensor_bytes(tensor) for tensor in features.values()), default=tensor_bytes(logits))
    total_bytes = parameter_bytes(model) + activation_peak
    return {"peak_bytes": float(total_bytes), "peak_mb": float(total_bytes) / (1024**2)}
