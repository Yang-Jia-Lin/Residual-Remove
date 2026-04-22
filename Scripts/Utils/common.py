"""Scripts/common.py
   实验脚本的公共初始化（参数解析、配置加载、模型构建、数据集准备）
"""
import argparse
import random
from typing import Any

import torch

from Configs.paras import DATA_DIR
from Configs.model_config import model_config
from Src.Models_Nets import build_model
from Src.DatasetsProcess.datasets import make_dataloaders


# ── 根据数据集名称自动推断合理的图片尺寸 ─────────────────────────────────────
_DEFAULT_IMAGE_SIZE: dict[str, int] = {
    "imagenet100": 224,
    "imagenet":    224,
    "cifar10":     32,
    "cifar100":    32,
    "synthetic":   32,
}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """向 parser 添加所有实验脚本通用的命令行参数"""
    # ── 环境参数 ────────────────────────────────────────────────────────
    parser.add_argument(
        "--device", default=None,
        help=f"运行设备，如 'cuda:0' 或 'cpu'。不指定则使用 model_config 中的值（当前：{model_config.hardware.device}）"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=f"随机种子。不指定则使用 model_config 中的值（当前：{model_config.hardware.seed}）"
    )

    # ── 模型参数 ────────────────────────────────────────────────────────
    parser.add_argument(
        "--model", default="resnet50",
        help="模型名称，如 resnet18 / resnet50 / mobilenet_v2"
    )
    parser.add_argument(
        "--pretrained", action="store_true", default=True,
        help="是否加载 ImageNet 预训练权重（默认：True）"
    )
    parser.add_argument(
        "--no-pretrained", dest="pretrained", action="store_false",
        help="从随机初始化开始（用于消融对照）"
    )

    # ── 数据集参数 ──────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset", default=None,
        help=f"数据集名称。不指定则使用 model_config 中的值（当前：{model_config.data.default_dataset}）"
    )
    parser.add_argument(
        "--data-root", default=None,
        help=f"数据集根目录。不指定则使用 DATA_DIR（当前：{DATA_DIR}）"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=f"batch size。不指定则使用 model_config 中的值（当前：{model_config.train.batch_size}）"
    )
    parser.add_argument(
        "--val-size", type=int, default=None,
        help="验证集最多使用的样本数，不指定则使用完整验证集。"
    )
    parser.add_argument(
        "--image-size", type=int, default=None,
        help="输入图片尺寸。不指定则根据数据集自动推断（imagenet100=224，cifar=32）。"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help=f"DataLoader worker 数量。不指定则使用 model_config 中的值（当前：{model_config.hardware.num_workers}）"
    )

    # ── 实验控制 ────────────────────────────────────────────────────────
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="每次 evaluate_model 最多跑多少个 batch。"
             "不指定则跑完整个验证集。调试时可设为 10 快速验证流程。"
    )


def build_setup(
    args: argparse.Namespace,
    compensator_name: str = "identity",
    compensator_rank: int = 16,
) -> dict[str, Any]:
    """根据命令行参数和 model_config，完成模型、数据集、设备的初始化"""
    
    # ── 设备 ────────────────────────────────────────────────────────────
    device_str = args.device or model_config.hardware.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"[setup] ⚠ 指定了 {device_str} 但 CUDA 不可用，自动降级到 cpu")
        device_str = "cpu"
    device = torch.device(device_str)

    # ── 随机种子 ────────────────────────────────────────────────────────
    seed = args.seed if args.seed is not None else model_config.hardware.seed
    torch.manual_seed(seed)
    random.seed(seed)

    # ── 数据集参数 ──────────────────────────────────────────────────────
    dataset_name = args.dataset   or model_config.data.default_dataset
    data_root    = args.data_root or DATA_DIR
    num_workers  = args.num_workers  if args.num_workers  is not None else model_config.hardware.num_workers
    batch_size   = args.batch_size   if args.batch_size   is not None else model_config.train.batch_size
    image_size   = (
        args.image_size
        or _DEFAULT_IMAGE_SIZE.get(dataset_name.lower())
        or model_config.data.default_image_size
    )

    # ── 构建数据集 ──────────────────────────────────────────────────────
    bundle = make_dataloaders(
        dataset_name         = dataset_name,
        data_root            = data_root,
        batch_size           = batch_size,
        image_size           = image_size,
        num_workers          = num_workers,
        synthetic_if_missing = True,
        val_size             = args.val_size,
        seed                 = seed,
    )

    # ── 构建模型 ────────────────────────────────────────────────────────
    model = build_model(
        model_name       = args.model,
        num_classes      = bundle.num_classes,
        pretrained       = args.pretrained,
        compensator_name = compensator_name,
        compensator_rank = compensator_rank,
    ).to(device)
    model.eval()
    n_blocks = len(model.get_block_names())

    print(f"[setup] 设备：{device}  种子：{seed}")
    print(f"[setup] 数据集：{dataset_name}  图片尺寸：{image_size}×{image_size}  类别数：{bundle.num_classes}")
    print(f"[setup] batch_size：{batch_size}  验证批次数：{len(bundle.val_loader)}  num_workers：{num_workers}")
    print(f"[setup] 模型：{args.model}  残差块数：{n_blocks}  预训练：{args.pretrained}")

    return {"model": model, "bundle": bundle, "device": device}


def get_probe_batch(
    bundle,
    device: torch.device,
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从验证集取一个 batch，用于探测中间特征的形状和大小。

    专为"先跑一次前向、观察中间张量"的实验设计（如 run_residual_stats.py）。
    batch_size 默认为 1，因为通常只需要形状信息。
    """
    images, labels = next(iter(bundle.val_loader))
    if images.size(0) > batch_size:
        images = images[:batch_size]
        labels = labels[:batch_size]
    return images.to(device), labels.to(device)


def resolve_removed_blocks(
    removed_blocks_arg: str,
    all_block_names: list[str],
) -> list[str]:
    """解析命令行传入的 removed_blocks 参数，转换为实际要删除的块名称列表。

    Args:
        removed_blocks_arg: 'all' | 'none' | 逗号分隔的块名，如 'layer1.0,layer2.1'
        all_block_names:    模型中所有残差块的名称列表

    Returns:
        实际要删除的残差块名称列表
    """
    if not removed_blocks_arg or removed_blocks_arg.lower() == "none":
        return []

    if removed_blocks_arg.lower() == "all":
        return list(all_block_names)

    selected = [b.strip() for b in removed_blocks_arg.split(",") if b.strip()]
    invalid  = [b for b in selected if b not in all_block_names]
    if invalid:
        raise ValueError(
            f"错误：指定的残差块不存在于模型中 -> {invalid}\n"
            f"当前模型支持的块：{all_block_names}"
        )
    return selected