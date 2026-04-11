from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models.builder import build_model
from src.utils.config import load_yaml
from src.utils.datasets import DatasetBundle, make_dataloaders
from src.utils.runtime import resolve_device, set_seed


def load_project_configs() -> dict[str, dict]:
    return {
        "env": load_yaml("configs/default_env.yaml"),
        "models": load_yaml("configs/models.yaml"),
        "compensator": load_yaml("configs/compensator.yaml"),
        "system": load_yaml("configs/system.yaml"),
    }


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    configs = load_project_configs()
    env_cfg = configs["env"]
    parser.add_argument("--dataset", default=env_cfg["default_dataset"])
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--device", default=env_cfg["device"])
    parser.add_argument("--data-root", default=env_cfg["data_root"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=env_cfg["seed"])
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--adapter-activation", default="gelu")
    return parser


def build_setup(args, compensator_name: str = "identity") -> dict:
    configs = load_project_configs()
    env_cfg = configs["env"]
    model_cfg = configs["models"][args.model]
    set_seed(args.seed)

    input_size = args.input_size or int(model_cfg["input_size"])
    num_classes = args.num_classes or int(model_cfg["num_classes"])
    width_mult = float(model_cfg.get("width_mult", 1.0))
    device = resolve_device(args.device)
    bundle: DatasetBundle = make_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=input_size,
        num_classes=num_classes,
        num_workers=int(env_cfg["num_workers"]),
        synthetic_if_missing=bool(env_cfg["synthetic_if_missing"]),
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    model = build_model(
        model_name=args.model,
        num_classes=bundle.num_classes,
        input_size=input_size,
        compensator_name=compensator_name,
        compensator_rank=args.rank,
        adapter_activation=args.adapter_activation,
        width_mult=width_mult,
    ).to(device)
    return {
        "configs": configs,
        "bundle": bundle,
        "model": model,
        "device": device,
        "input_size": input_size,
        "num_classes": bundle.num_classes,
    }


def resolve_removed_blocks(spec: str | None, all_blocks: list[str]) -> list[str]:
    if not spec or spec == "all":
        return list(all_blocks)
    if spec.startswith("last:"):
        count = max(0, int(spec.split(":", 1)[1]))
        return list(all_blocks[-count:]) if count > 0 else []
    if spec.startswith("first:"):
        count = max(0, int(spec.split(":", 1)[1]))
        return list(all_blocks[:count])
    names = [item.strip() for item in spec.split(",") if item.strip()]
    return [name for name in all_blocks if name in names]


def get_probe_batch(bundle: DatasetBundle, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = next(iter(bundle.val_loader))
    return images.to(device), labels.to(device)
