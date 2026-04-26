"""Scripts/Utils/run_finetune.py
python -m Scripts.Utils.run_finetune --model resnet50 --epochs 10
python -m Scripts.Utils.run_finetune --model resnet18 --epochs 5 --lr 5e-4
"""

import argparse

from Scripts.Utils.script_common import add_common_args, build_setup
from Src.Training.finetune import finetune_head


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Fine-tune 分类头")
        add_common_args(parser)
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--lr", type=float, default=1e-3)
        args = parser.parse_args()

    setup = build_setup(args)
    model, bundle, device = setup["model"], setup["bundle"], setup["device"]

    finetune_head(
        model=model,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_name=f"{args.model}_{args.dataset}.pth",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune 分类头")
    add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # 自定义
    args.model = "resnet18"
    main(args)
