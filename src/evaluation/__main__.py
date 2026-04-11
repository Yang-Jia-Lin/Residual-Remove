# src/evaluation/__main__.py
"""一键运行所有评估模块的冒烟测试。

运行方式：python -m src.evaluation
"""
from __future__ import annotations

import yaml
from pathlib import Path

import torch

from models import build_model, get_block_names
from src.evaluation.flops   import analyze_model
from src.evaluation.latency import compare_latency
from src.evaluation.memory  import compare_memory


def main() -> None:
    # ── 加载配置 ──────────────────────────────────────────────────────────
    cfg_path = Path("_configs/default_env.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    device     = torch.device(cfg["device"])
    image_size = cfg["data"]["default_image_size"]
    seed       = cfg["seed"]
    torch.manual_seed(seed)

    print(f"配置加载自：{cfg_path}")
    print(f"  device     : {device}")
    print(f"  image_size : {image_size}\n")

    # ── 构建模型 ──────────────────────────────────────────────────────────
    # 用最轻量的组合：ResNet-18 + IdentityCompensator
    # 冒烟测试不需要好的精度，只需要流程跑通
    model = build_model("resnet18", num_classes=10, pretrained=False).to(device)
    model.eval()

    # 构造一张假图，shape = (1, 3, H, W)
    # batch_size=1 是延迟测量的标准场景（单次推理）
    sample = torch.randn(1, 3, image_size, image_size, device=device)

    print("=" * 55)
    print("① FLOPs / 参数量分析")
    print("=" * 55)
    cost = analyze_model(model, sample, mode="full")
    print(cost)

    print("\n" + "=" * 55)
    print("② 延迟对比（full vs plain）")
    print("=" * 55)
    # 冒烟测试减少重复次数，加快运行速度
    latency = compare_latency(model, sample, repetitions=10, warmup=3)
    print(latency)

    # 验证加速比是正数（不验证大小，因为合成模型行为不稳定）
    assert latency.speedup > 0, "加速比应为正数"

    print("\n" + "=" * 55)
    print("③ 峰值内存对比（full vs plain）")
    print("=" * 55)
    memory = compare_memory(model, sample)
    print(memory)

    # 验证内存数字是有限正数
    assert memory.full.peak_mb  >= 0, "full mode 峰值内存不应为负"
    assert memory.plain.peak_mb >= 0, "plain mode 峰值内存不应为负"

    print("\n✓ 所有评估模块冒烟测试通过。")


if __name__ == "__main__":
    main()