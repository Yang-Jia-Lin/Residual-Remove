from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from residual_remove.models.resnet_ablation import ResidualBlockInfo


def summarize_block_infos(block_infos: Iterable[ResidualBlockInfo]) -> Dict[str, object]:
    infos = list(block_infos)
    total = len(infos)
    kept = sum(1 for info in infos if info.keep_residual)
    removed = total - kept
    by_stage: Dict[str, Dict[str, int]] = {}
    for info in infos:
        stage_key = f"stage{info.stage}"
        stage_stats = by_stage.setdefault(stage_key, {"kept": 0, "removed": 0})
        stage_stats["kept" if info.keep_residual else "removed"] += 1
    return {
        "total_blocks": total,
        "kept_blocks": kept,
        "removed_blocks": removed,
        "removed_ratio": removed / total if total else 0.0,
        "by_stage": by_stage,
    }


def calculate_activation_lifetime_proxy(
    model: nn.Module,
    sample_batch: torch.Tensor,
) -> Dict[str, float]:
    stats = {
        "kept_residual_elements": 0.0,
        "lifetime_proxy_elements_x_ops": 0.0,
    }

    hooks = []

    def make_hook(module: nn.Module):
        def hook(_module: nn.Module, inputs, _output):
            if not getattr(module, "keep_residual", False):
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            numel = float(x.numel())
            stats["kept_residual_elements"] += numel
            stats["lifetime_proxy_elements_x_ops"] += numel * float(
                getattr(module, "lifetime_ops", 0)
            )

        return hook

    for module in model.modules():
        if isinstance(module, (BasicBlock, Bottleneck)):
            hooks.append(module.register_forward_hook(make_hook(module)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(sample_batch)
    if was_training:
        model.train()

    for hook in hooks:
        hook.remove()

    return stats


def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    sample_batch: torch.Tensor,
    warmup_steps: int = 10,
    measure_steps: int = 30,
) -> Dict[str, float]:
    model.eval()
    sample_batch = sample_batch.to(device, non_blocking=True)

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup_steps):
            with torch.no_grad():
                model(sample_batch)
        torch.cuda.synchronize()

        timings_ms: List[float] = []
        for _ in range(measure_steps):
            starter.record()
            with torch.no_grad():
                model(sample_batch)
            ender.record()
            torch.cuda.synchronize()
            timings_ms.append(starter.elapsed_time(ender))
        mean_latency_ms = sum(timings_ms) / len(timings_ms)
    else:
        for _ in range(warmup_steps):
            with torch.no_grad():
                model(sample_batch)
        timings_ms = []
        for _ in range(measure_steps):
            start = time.perf_counter()
            with torch.no_grad():
                model(sample_batch)
            end = time.perf_counter()
            timings_ms.append((end - start) * 1000.0)
        mean_latency_ms = sum(timings_ms) / len(timings_ms)

    batch_size = int(sample_batch.shape[0])
    throughput = batch_size / (mean_latency_ms / 1000.0)
    return {
        "latency_ms": mean_latency_ms,
        "throughput_samples_per_sec": throughput,
    }


class AblationProfiler:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def profile_peak_memory(
        self,
        model: nn.Module,
        sample_batch: torch.Tensor,
        criterion: nn.Module | None = None,
    ) -> Dict[str, float]:
        if self.device.type != "cuda":
            return {"peak_memory_mb": 0.0, "peak_memory_note": "CUDA unavailable"}

        model.train()
        criterion = criterion or nn.CrossEntropyLoss()
        sample_batch = sample_batch.to(self.device, non_blocking=True)
        labels = torch.zeros(sample_batch.size(0), dtype=torch.long, device=self.device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        outputs = model(sample_batch)
        loss = criterion(outputs, labels)
        loss.backward()
        peak_bytes = torch.cuda.max_memory_allocated(self.device)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        return {"peak_memory_mb": peak_bytes / (1024 ** 2)}

    @staticmethod
    def save_report(output_dir: str, report: Dict[str, object], block_infos: List[ResidualBlockInfo]) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        serializable = dict(report)
        serializable["block_infos"] = [asdict(info) for info in block_infos]

        with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2, ensure_ascii=False)
