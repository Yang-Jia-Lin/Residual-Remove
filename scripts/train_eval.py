from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from residual_remove.models import build_resnet_with_ablation
from residual_remove.utils import (
    AblationProfiler,
    benchmark_inference,
    build_dataloaders,
    calculate_activation_lifetime_proxy,
    evaluate,
    summarize_block_infos,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and profile residual ablation experiments.")
    parser.add_argument("--model", choices=["resnet18", "resnet50"], required=True)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ablation-mode",
        choices=["none", "random_ratio", "stage_progressive", "full"],
        default="none",
    )
    parser.add_argument("--ablation-value", type=float, default=0.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, num_classes = build_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model, block_infos = build_resnet_with_ablation(
        model_name=args.model,
        num_classes=num_classes,
        ablation_mode=args.ablation_mode,
        ablation_value=args.ablation_value,
        seed=args.seed,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        epoch_metrics = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

    sample_images, _ = next(iter(test_loader))
    sample_batch = sample_images[: min(args.batch_size, 32)]
    profiler = AblationProfiler(device)

    report = {
        "model": args.model,
        "dataset": args.dataset,
        "device": str(device),
        "epochs": args.epochs,
        "ablation": {
            "mode": args.ablation_mode,
            "value": args.ablation_value,
        },
        "block_summary": summarize_block_infos(block_infos),
        "final_train_eval": history[-1] if history else {},
        "latency": benchmark_inference(model, device, sample_batch),
        "activation_lifetime_proxy": calculate_activation_lifetime_proxy(
            model,
            sample_batch.to(device),
        ),
        "peak_memory": profiler.profile_peak_memory(model, sample_batch, criterion),
        "history": history,
    }
    profiler.save_report(args.output_dir, report, block_infos)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
