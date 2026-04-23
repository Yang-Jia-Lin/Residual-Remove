"""Scripts/common.py
   实验脚本的公共初始化（参数解析、配置加载、模型构建、数据集准备）
"""
import argparse
import random
from typing import Any

import torch
from pathlib import Path

from Configs.paras import RESULT_DIR
from Configs.paras import DATA_DIR
from Configs.model_config import model_config
from Src.Models_Nets import build_model
from Src.Models_Training.finetune import load_finetuned
from Src.Utils.data_utils import make_dataloaders


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
        "--device", default=model_config.hardware.device,
        help="运行设备，如 'cuda:0' 或 'cpu'"
    )
    parser.add_argument(
        "--seed", type=int, default=model_config.hardware.seed,
        help="随机种子"
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
    parser.add_argument(
        "--checkpoint", default="auto",
        help="fine-tuned checkpoint 路径或文件名。设为 'auto' 会自动寻找 {model}_{dataset}.pth，不指定则用默认权重。"
    )

    # ── 数据集参数 ──────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset", default=model_config.data.default_dataset,
        help="数据集名称"
    )
    parser.add_argument(
        "--data-root", default=DATA_DIR,  # <--- 直接使用 paras.py 中的 DATA_DIR
        help="数据集根目录"
    )
    parser.add_argument(
        "--batch-size", type=int, default=model_config.train.batch_size,
        help="batch size"
    )
    parser.add_argument(
        "--val-size", type=int, default=None,
        help="验证集最多使用的样本数，不指定则使用完整验证集。"
    )
    parser.add_argument(
        "--image-size", type=int, default=None,
        help="输入图片尺寸。不指定则根据数据集自动推断或使用默认值。"
    )
    parser.add_argument(
        "--num-workers", type=int, default=model_config.hardware.num_workers,
        help="DataLoader worker 数量"
    )

    # ── 实验控制 ────────────────────────────────────────────────────────
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="每次 evaluate_model 最多跑多少个 batch。不指定则跑完整个验证集。调试时可设为 10 快速验证流程。"
    )

def build_setup(
    args: argparse.Namespace,
    compensator_name: str = "identity",
    compensator_rank: int = 16,
) -> dict[str, Any]:
    """根据命令行参数和 model_config，完成模型、数据集、设备的初始化"""
    
    # ── 1. 全局配置同步 (Single Source of Truth) ────────────────────────
    # 将解析到的命令行参数写回 model_config，确保全局环境获取到的参数一致
    model_config.hardware.device = args.device
    model_config.hardware.seed = args.seed
    model_config.hardware.num_workers = args.num_workers
    model_config.data.default_dataset = args.dataset
    model_config.train.batch_size = args.batch_size

    # ── 设备与随机种子 ──────────────────────────────────────────────────
    device_str = model_config.hardware.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"[setup] ⚠ 指定了 {device_str} 但 CUDA 不可用，自动降级到 cpu")
        device_str = "cpu"
        model_config.hardware.device = "cpu"  # 同步降级信息
    
    device = torch.device(device_str)
    
    seed = model_config.hardware.seed
    torch.manual_seed(seed)
    random.seed(seed)

    # ── 数据集参数推断 ──────────────────────────────────────────────────
    # 如果命令行没传 image_size，优先看字典推断，最后用 model_config 兜底
    final_image_size = (
        args.image_size
        or _DEFAULT_IMAGE_SIZE.get(model_config.data.default_dataset.lower())
        or model_config.data.default_image_size
    )
    model_config.data.default_image_size = final_image_size # 同步最终推断结果

    # ── 构建数据集 ──────────────────────────────────────────────────────
    bundle = make_dataloaders(
        dataset_name         = model_config.data.default_dataset,
        data_root            = args.data_root,
        batch_size           = model_config.train.batch_size,
        image_size           = final_image_size,
        num_workers          = model_config.hardware.num_workers,
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
    

    # ── 自动/手动加载 checkpoint ──
    if hasattr(args, "checkpoint") and args.checkpoint is not None:
        ckpt_dir = RESULT_DIR / "Checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True) # 确保文件夹存在

        if args.checkpoint == "auto":
            # 自动推断权重文件名，例如: resnet50_imagenet100.pth
            auto_ckpt_name = f"{args.model}_{model_config.data.default_dataset}.pth"
            ckpt_path = ckpt_dir / auto_ckpt_name
            
            if ckpt_path.exists():
                print(f"[setup] 自动检测到微调权重，正在加载: {auto_ckpt_name}")
                load_finetuned(model, str(ckpt_path))
            else:
                print(f"[setup] 未找到 {auto_ckpt_name}，使用原始 pretrained 权重。")
        else:
            # 兼容手动指定权重的情况
            ckpt_path = Path(args.checkpoint) if Path(args.checkpoint).is_absolute() else ckpt_dir / args.checkpoint
            if ckpt_path.exists():
                print(f"[setup] 手动指定加载权重: {ckpt_path.name}")
                load_finetuned(model, str(ckpt_path))
            else:
                raise FileNotFoundError(f"找不到指定的权重文件: {ckpt_path}")
    n_blocks = len(model.get_block_names())

    print(f"[setup] 设备：{device}  种子：{seed}")
    print(f"[setup] 数据集：{model_config.data.default_dataset}  图片尺寸：{final_image_size}×{final_image_size}  类别数：{bundle.num_classes}")
    print(f"[setup] batch_size：{model_config.train.batch_size}  验证批次数：{len(bundle.val_loader)}  num_workers：{model_config.hardware.num_workers}")
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