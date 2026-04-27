"""Scripts/common.py
实验脚本的公共初始化（参数解析、配置加载、模型构建、数据集准备）
"""

import argparse
import random
from pathlib import Path
from typing import Any

import torch

from Configs.model_config import model_config
from Configs.paras import DATA_DIR, RESULT_DIR
from Src.ModelNets.builder import build_model
from Src.Training.finetune import load_finetuned
from Src.Utils.data_utils import make_dataloaders

_DEFAULT_IMAGE_SIZE: dict[str, int] = {
    "imagenet100": 224,
    "imagenet": 224,
    "cifar10": 32,
    "cifar100": 32,
}


def get_probe_batch(
    bundle,
    device: torch.device,
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从验证集取一个 batch，探测中间特征的形状和大小"""
    images, labels = next(iter(bundle.val_loader))
    if images.size(0) > batch_size:
        images = images[:batch_size]
        labels = labels[:batch_size]
    return images.to(device), labels.to(device)


def resolve_removed_blocks(
    removed_blocks_arg: str,
    all_block_names: list[str],
) -> list[str]:
    """解析命令行传入的 removed_blocks 参数，转换为实际要删除的块名称列表"""
    if not removed_blocks_arg or removed_blocks_arg.lower() == "none":
        return []

    if removed_blocks_arg.lower() == "all":
        return list(all_block_names)

    selected = [b.strip() for b in removed_blocks_arg.split(",") if b.strip()]
    invalid = [b for b in selected if b not in all_block_names]
    if invalid:
        raise ValueError(
            f"错误：指定的残差块不存在于模型中 -> {invalid}\n"
            f"当前模型支持的块：{all_block_names}"
        )
    return selected


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """通用的命令行参数"""
    # ── 环境参数 ────────────────────────────────────────────────────────
    parser.add_argument(
        "--device",
        default=model_config.hardware.device,
        help="运行设备，如 'cuda:0' 或 'cpu'",
    )
    parser.add_argument(
        "--seed", type=int, default=model_config.hardware.seed, help="随机种子"
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=model_config.hardware.max_memory_gb,
        help="限制 GPU 显存上限（GB），用于模拟小显存设备。不指定则使用全部显存。",
    )

    # ── 模型参数 ────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        default="resnet50",
        help="模型名称，如 resnet18 / resnet50 / mobilenet_v2（默认：resnet50）",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="是否加载预训练权重（默认：True）",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="从随机初始化开始（存在这个命令则覆盖上述 --pretrained 的默认值 True）",
    )
    parser.add_argument(
        "--checkpoint",
        default="auto",
        help="fine-tuned checkpoint 路径或文件名。设为 'auto' 会自动寻找 {model}_{dataset}.pth，不指定则用默认权重。",
    )

    # ── 数据集参数 ──────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset",
        default=model_config.data.default_dataset,
        help="数据集名称，默认从 model_config 读取（imagenet100）",
    )
    parser.add_argument(
        "--data-root",
        default=DATA_DIR,  # <--- 直接使用 paras.py 中的 DATA_DIR
        help="数据集根目录",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=model_config.train.batch_size,
        help="batch size，默认从 model_config 读取（32）",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help="验证集最多使用的样本数，不指定则使用完整验证集",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="输入图片尺寸。不指定则根据数据集自动推断或使用默认值",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=model_config.hardware.num_workers,
        help="DataLoader worker 数量，默认从 model_config 读取（4）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出结果路径",
    )

    # ── 实验控制 ────────────────────────────────────────────────────────
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="每次 evaluate_model 最多跑多少个 batch。不指定则跑完整个验证集。调试时可设为 10 快速验证流程。",
    )


def build_setup(
    args: argparse.Namespace,
    compensator_name: str = "identity",
    compensator_rank: int = 16,
) -> dict[str, Any]:
    """模型、数据集、设备初始化"""

    # 1. 全局参数
    model_config.hardware.device = args.device
    model_config.hardware.seed = args.seed
    model_config.hardware.num_workers = args.num_workers
    model_config.data.default_dataset = args.dataset
    model_config.train.batch_size = args.batch_size

    device_str = model_config.hardware.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"[⚠setup] 指定了 {device_str} 但 CUDA 不可用，自动降级到 cpu")
        device_str = "cpu"
        model_config.hardware.device = "cpu"  # 同步降级信息
    device = torch.device(device_str)
    seed = model_config.hardware.seed
    torch.manual_seed(seed)
    random.seed(seed)

    final_image_size = (
        args.image_size
        or _DEFAULT_IMAGE_SIZE.get(model_config.data.default_dataset.lower())
        or model_config.data.default_image_size
    )
    model_config.data.default_image_size = final_image_size

    # 2. 数据集
    bundle = make_dataloaders(
        dataset_name=model_config.data.default_dataset,
        data_root=args.data_root,
        batch_size=model_config.train.batch_size,
        image_size=final_image_size,
        num_workers=model_config.hardware.num_workers,
    )

    # 3. 模型
    model = build_model(
        model_name=args.model,
        num_classes=bundle.num_classes,
        pretrained=args.pretrained,
        compensator_name=compensator_name,
        compensator_rank=compensator_rank,
    ).to(device)
    model.eval()

    # 4. 模型 checkpoint
    if hasattr(args, "checkpoint") and args.checkpoint is not None:
        ckpt_dir = RESULT_DIR / "Checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if args.checkpoint == "auto":
            auto_ckpt_name = f"{args.model}_{model_config.data.default_dataset}.pth"
            ckpt_path = ckpt_dir / auto_ckpt_name
            if ckpt_path.exists():
                load_finetuned(model, str(ckpt_path))
            else:
                print(f"[⚠setup] 未找到 {auto_ckpt_name}，使用原始 pretrained 权重。")
        else:
            ckpt_path = (
                Path(args.checkpoint)
                if Path(args.checkpoint).is_absolute()
                else ckpt_dir / args.checkpoint
            )
            if ckpt_path.exists():
                load_finetuned(model, str(ckpt_path))
            else:
                raise FileNotFoundError(f"找不到指定的权重文件: {ckpt_path}")
    n_blocks = len(model.get_block_names())

    print(
        f"[setup-hardw]\t{device}\t\tnum_workers：{model_config.hardware.num_workers}\t\t种子：{seed}"
    )
    print(
        f"[setup-model]\t{args.model}\t残差块数：{n_blocks}\t\t预训练：{args.pretrained}"
    )
    print(
        f"[setup- data]\t{model_config.data.default_dataset}\t图片尺寸：{final_image_size}×{final_image_size}\t类别数：{bundle.num_classes}"
    )
    print(
        f"[setup-batch]\t{model_config.train.batch_size}\t\t验证批次：{len(bundle.val_loader)}"
    )

    return {"model": model, "bundle": bundle, "device": device}
