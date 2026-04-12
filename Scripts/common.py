"""experiments/common.py — 实验脚本的公共初始化逻辑。

所有 Exp1/Exp2/Exp3 脚本都通过 add_common_args + build_setup 完成
参数解析、配置加载、模型构建、数据集准备这四件事，
避免每个脚本重复写同样的样板代码。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
import yaml

from Src.Models_Nets import build_model
from Src.Utils.datasets import make_dataloaders


# 配置文件的默认位置，相对于项目根目录
_DEFAULT_CONFIG = Path("Configs/default_env.yaml")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """向 parser 添加所有实验脚本通用的命令行参数。
    
    设计原则：每个参数都有合理的默认值，最小化运行一个实验所需的
    手动指定参数数量。同时，关键参数（device、dataset）可以通过
    命令行覆盖 yaml，方便在不同环境下切换而不用修改配置文件。
    """
    # ── 环境参数（可覆盖 yaml）──────────────────────────────────────────
    parser.add_argument(
        "--config", default=str(_DEFAULT_CONFIG),
        help="yaml 配置文件路径（默认：_configs/default_env.yaml）"
    )
    parser.add_argument(
        "--device", default=None,
        help="运行设备，如 'cuda:0' 或 'cpu'。不指定则使用 yaml 中的值。"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子。不指定则使用 yaml 中的值。"
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
        help="数据集名称，如 imagenet / cifar100。不指定则使用 yaml 中的值。"
    )
    parser.add_argument(
        "--data-root", default=None,
        help="数据集根目录。不指定则使用 yaml 中的值。"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="评估时的 batch size（默认：64）"
    )
    parser.add_argument(
        "--val-size", type=int, default=None,
        help="验证集最多使用的样本数，不指定则使用完整验证集。"
    )
    parser.add_argument(
        "--image-size", type=int, default=None,
        help="输入图片尺寸。不指定则根据数据集自动选择（imagenet=224，cifar=32）。"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="DataLoader 的 worker 数量。不指定则使用 yaml 中的值。"
    )

    # ── 实验控制 ────────────────────────────────────────────────────────
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="每次 evaluate_model 最多跑多少个 batch。"
             "不指定则跑完整个验证集。调试时可以设为 10 快速验证流程。"
    )


# 根据数据集名称自动推断合理的图片尺寸
_DEFAULT_IMAGE_SIZE: dict[str, int] = {
    "imagenet": 224,
    "cifar10":  32,
    "cifar100": 32,
    "synthetic": 32,
}


def build_setup(
    args: argparse.Namespace,
    compensator_name: str = "identity",
    compensator_rank: int = 16,
) -> dict[str, Any]:
    """根据命令行参数和 yaml 配置，完成模型、数据集、设备的初始化。

    优先级：命令行参数 > yaml 配置文件 > 函数内硬编码默认值。
    
    返回一个包含以下 key 的字典：
        model   : InjectedModel，已移到正确设备上，处于 eval 模式
        bundle  : DatasetBundle，包含 train_loader 和 val_loader
        device  : torch.device
        cfg     : 原始 yaml 字典，供需要访问其他配置项的脚本使用
    """
    # ── 加载 yaml 配置 ──────────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    

    # ── 解析设备（命令行优先） ───────────────────────────────────────────
    device_str = args.device or cfg.get("device", "cpu")
    # 如果指定了 cuda 但实际不可用，自动降级到 cpu 并警告
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"[setup] ⚠ 指定了 {device_str} 但 CUDA 不可用，自动降级到 cpu")
        device_str = "cpu"
    device = torch.device(device_str)

    # ── 设置随机种子 ────────────────────────────────────────────────────
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    torch.manual_seed(seed)
    random.seed(seed)
    # print(f"[setup] 随机种子：{seed}")

    # ── 解析数据集参数 ──────────────────────────────────────────────────
    dataset_name = args.dataset or cfg["data"]["default_dataset"]
    data_root    = args.data_root or cfg["paths"]["data_root"]
    num_workers  = args.num_workers if args.num_workers is not None else cfg.get("num_workers", 0)

    # 图片尺寸：命令行 > 自动推断（根据数据集名称）
    image_size = (
        args.image_size
        or _DEFAULT_IMAGE_SIZE.get(dataset_name.lower())
        or cfg["data"].get("default_image_size", 32)
    )
    
    # ── 构建数据集 ──────────────────────────────────────────────────────
    bundle = make_dataloaders(
        dataset_name         = dataset_name,
        data_root            = data_root,
        batch_size           = args.batch_size,
        image_size           = image_size,
        num_workers          = num_workers,
        synthetic_if_missing = cfg["data"].get("synthetic_if_missing", True),
        val_size             = args.val_size,   # None 表示使用完整验证集
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

    print(f"配置加载自：{cfg_path}")
    print(f"[setup] 运行设备：{device}")
    print(f"[setup] 数据集：{dataset_name}，图片尺寸：{image_size}×{image_size}，类别数：{bundle.num_classes}")
    print(f"[setup] batch_size：{args.batch_size}，验证集批次数：{len(bundle.val_loader)}，num_workers：{num_workers}") 
    print(f"[setup] 模型：{args.model}，残差块数：{n_blocks}，预训练：{args.pretrained}")

    return {"model": model, "bundle": bundle, "device": device, "cfg": cfg}


def get_probe_batch(
    bundle,
    device: torch.device,
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从验证集取一个 batch，用于探测中间特征的形状和大小。

    这个函数专门为需要"先跑一次前向、观察中间张量"的实验设计，
    比如 run_residual_stats.py 和 run_system_cost.py。
    batch_size=1 是默认值，因为我们只需要形状信息，不需要统计意义上的均值。
    """
    # 从 val_loader 取第一个 batch，忽略后续所有 batch
    images, labels = next(iter(bundle.val_loader))

    # 如果 val_loader 的 batch_size 比需要的大，只取前 batch_size 张
    if images.size(0) > batch_size:
        images = images[:batch_size]
        labels = labels[:batch_size]

    return images.to(device), labels.to(device)