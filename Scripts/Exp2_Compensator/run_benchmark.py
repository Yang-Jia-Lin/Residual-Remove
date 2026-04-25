"""Scripts/Exp2_Compensator/run_benchmark.py"""
import argparse
import logging
import sys
import torch
from datetime import datetime
from pathlib import Path

from Configs.paras import RESULT_DIR_2
from Configs.model_config import model_config
from Scripts.Utils.script_common import add_common_args, build_setup, resolve_removed_blocks
from Src.Metrics.flops import count_parameters, estimate_macs
from Src.Metrics.latency import measure_latency
from Src.Metrics.memory import measure_peak_memory
from Src.Training.calibrate import calibrate_compensators
from Src.Metrics.accuracy import evaluate_model
from Src.Utils.data_utils import build_calibration_loader
from Src.Utils.runtime import write_csv


# ---------------------------------------------------------------------------
# 补偿方法列表：(输出名称, compensator_name, compensator_rank)
# rank=0 的条目不使用 rank 参数（scalar / affine / linear1x1 内部不需要）
# ---------------------------------------------------------------------------
VARIANTS: list[tuple[str, str, int]] = [
    # ("scalar",        "scalar",    0),
    # ("affine",        "affine",    0),
    ("linear1x1",     "linear1x1", 0),
    # ("low_rank_r4",   "low_rank",  4),
    ("low_rank_r8",   "low_rank",  8),
    ("low_rank_r16",  "low_rank",  16),
    ("low_rank_r32",  "low_rank",  32),
    ("adapter",       "adapter",   16),
]

OUTPUT_ROOT = RESULT_DIR_2

# linear1x1 参数量是其他方法的 ~80 倍（2048×2048），
# 用默认 lr=1e-3 步子太大，单独指定小 lr
VARIANT_LR: dict[str, float] = {
    "linear1x1": 1e-4,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="实验2：对比各补偿方法在相同校准设置下的综合指标"
    )
    add_common_args(parser)
    parser.add_argument(
        "--removed-blocks", default="layer2.3",
        # Resnet50 支持的块：
        # ['layer1.0', 'layer1.1', 'layer1.2', 
        # 'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3', 
        # 'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 
        # 'layer4.0', 'layer4.1', 'layer4.2']
        help="要删除的残差块，'all' 表示全部（默认 all）",
    )
    parser.add_argument(
        "--calib-size", type=int, default=model_config.data.calib_num_samples,
        help=f"校准图像数量（默认 {model_config.data.calib_num_samples}）",
    )
    parser.add_argument(
        "--calib-batch-size", type=int, default=model_config.data.calib_batch_size,
        help=f"校准 DataLoader 的 batch size（默认 {model_config.data.calib_batch_size}）",
    )
    parser.add_argument(
        "--epochs", type=int, default=model_config.train.epochs, 
        help=f"补偿器校准轮数（默认 {model_config.train.epochs}）",
    )
    parser.add_argument(
        "--lr", type=float, default=model_config.train.lr, 
        help=f"补偿器学习率（默认 {model_config.train.lr}）",
    )
    parser.add_argument(
        "--latency-reps", type=int, default=20,
        help="延迟测量重复次数（默认 20）",
    )
    parser.add_argument(
        "--latency-warmup", type=int, default=5,
        help="延迟测量预热次数（默认 5）",
    )
    return parser

# ---------------------------------------------------------------------------
# 日志初始化：同时写文件和 stdout
# ---------------------------------------------------------------------------

def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("exp2")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
# 单条测量：给定已准备好的 model + sample，跑完所有指标
# ---------------------------------------------------------------------------

def _measure_all(
    model: torch.nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    mode: str,
    removed_blocks: list[str] | None,
    val_loader,
    latency_reps: int,
    latency_warmup: int,
    max_batches: int | None,
    logger: logging.Logger,
) -> dict:
    fwd_kwargs = dict(mode=mode, removed_blocks=removed_blocks)

    # 精度
    acc = evaluate_model(
        model, val_loader, device=device,
        mode=mode, removed_blocks=removed_blocks,
        max_batches=max_batches,
    )
    logger.info(f"    精度  top1={acc.top1:.2f}%  top5={acc.top5:.2f}%  loss={acc.loss:.4f}")

    # 延迟
    lat = measure_latency(
        model, sample,
        repetitions=latency_reps,
        warmup=latency_warmup,
        **fwd_kwargs,
    )
    logger.info(f"    延迟  均值={lat.mean_ms:.3f} ms  std=±{lat.std_ms:.3f} ms")

    # 峰值内存
    mem = measure_peak_memory(model, sample, **fwd_kwargs)
    logger.info(f"    内存  峰值={mem.peak_mb:.2f} MB  [{mem.method}]")

    # FLOPs & 参数量
    macs = estimate_macs(model, sample, **fwd_kwargs)
    params = count_parameters(model)
    logger.info(f"    规模  params={params:,}  MACs={macs:,}")

    return {
        "top1":          round(acc.top1,     4),
        "top5":          round(acc.top5,     4),
        "loss":          round(acc.loss,     6),
        "latency_ms":    round(lat.mean_ms,  4),
        "latency_std_ms":round(lat.std_ms,   4),
        "peak_memory_mb":round(mem.peak_mb,  2),
        "macs":          macs,
        "params":        params,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    args   = build_parser().parse_args()
    ts     = datetime.now().strftime("%Y%m%d_%H%M")
    outdir = OUTPUT_ROOT
    outdir.mkdir(parents=True, exist_ok=True)

    logger   = _setup_logger(outdir / f"{ts}.log")
    csv_path = outdir / f"{ts}_compensators_baseline.csv"

    logger.info("=" * 60)
    logger.info("实验2：补偿方法基准测试")
    logger.info(f"模型={args.model}  数据集={args.dataset}  设备={args.device}")
    logger.info(f"校准图像={args.calib_size}  轮数={args.epochs}  lr={args.lr}")
    logger.info("=" * 60)

    rows: list[dict] = []

    # ── 0. 准备公共资源（数据集 + 设备）────────────────────────────────────
    # 用 identity 补偿器加载一次，取 dataset 和 device，后面各 variant 复用
    base_setup = build_setup(args, compensator_name="identity")
    bundle     = base_setup["bundle"]
    device     = base_setup["device"]

    # 固定同一个 batch 作为延迟和内存的测量输入，保证各组条件完全一致
    sample, _ = next(iter(bundle.val_loader))
    sample     = sample.to(device)
    logger.info(f"测量 batch shape: {sample.shape}（固定，保证公平）\n")

    removed_blocks = resolve_removed_blocks(args.removed_blocks, base_setup["model"].get_block_names())
    logger.info(f"删除残差块数：{len(removed_blocks)}  ({removed_blocks[0]} → {removed_blocks[-1]})\n")

    # 校准数据集只建一次，各 variant 共享同一份随机子集（seed 固定，保证一致性）
    calib_loader = build_calibration_loader(
        bundle.train_dataset,
        calib_size=args.calib_size,
        batch_size=args.calib_batch_size,
        # 优先使用 args 中的 num_workers，如果没有则使用配置里的默认值
        num_workers=getattr(args, "num_workers", model_config.hardware.num_workers), # <--- 修改
        # 优先使用 args 中的 seed，如果没有则使用配置里的默认值
        seed=args.seed if getattr(args, "seed", None) is not None else model_config.hardware.seed, # <--- 修改
    )

    # ── 1. full_residual baseline ──────────────────────────────────────────
    logger.info("[0] full_residual（原始模型，保留所有残差）")
    full_model = base_setup["model"]
    metrics = _measure_all(
        full_model, sample, device,
        mode="full", removed_blocks=None,
        val_loader=bundle.val_loader,
        latency_reps=args.latency_reps,
        latency_warmup=args.latency_warmup,
        max_batches=getattr(args, "max_batches", None),
        logger=logger,
    )
    rows.append({
        "model": args.model, "dataset": bundle.source,
        "compensator": "full_residual", "rank": 0,
        "calib_size": 0, "epochs": 0,
        "final_calib_loss": 0.0,
        **metrics,
    })
    logger.info("")

    # ── 2. plain（暴力删残差，无补偿）────────────────────────────────────────
    logger.info("[1] plain（删除残差，无补偿）")
    plain_model = build_setup(args, compensator_name="identity")["model"]
    metrics = _measure_all(
        plain_model, sample, device,
        mode="plain", removed_blocks=removed_blocks,
        val_loader=bundle.val_loader,
        latency_reps=args.latency_reps,
        latency_warmup=args.latency_warmup,
        max_batches=getattr(args, "max_batches", None),
        logger=logger,
    )
    rows.append({
        "model": args.model, "dataset": bundle.source,
        "compensator": "plain", "rank": 0,
        "calib_size": 0, "epochs": 0,
        "final_calib_loss": 0.0,
        **metrics,
    })
    logger.info("")

    # ── 3. 各补偿方法 ─────────────────────────────────────────────────────
    for idx, (display_name, comp_name, comp_rank) in enumerate(VARIANTS, start=2):
        logger.info(f"[{idx}] {display_name}  (compensator={comp_name}, rank={comp_rank})")

        # 每个 variant 独立建模，避免参数污染
        setup = build_setup(
            args,
            compensator_name=comp_name,
            compensator_rank=comp_rank if comp_rank > 0 else 16,
        )
        model = setup["model"]

        # linear1x1 参数量大，单独用小 lr 防止训飞
        effective_lr = VARIANT_LR.get(display_name, args.lr)
        if effective_lr is None:
            effective_lr = model_config.train.lr
        logger.info(f"    校准中（{args.epochs} epochs, calib_size={args.calib_size}, lr={effective_lr}）...")
        history = calibrate_compensators(
            model,
            calibration_loader=calib_loader,
            device=device,
            removed_blocks=removed_blocks,
            epochs=args.epochs,
            lr=effective_lr,
            max_batches=getattr(args, "max_batches", None),
        )
        final_loss = history["epoch_loss"][-1]
        logger.info(f"    校准完成，最终 loss={final_loss:.6f}")

        metrics = _measure_all(
            model, sample, device,
            mode="compensated", removed_blocks=removed_blocks,
            val_loader=bundle.val_loader,
            latency_reps=args.latency_reps,
            latency_warmup=args.latency_warmup,
            max_batches=getattr(args, "max_batches", None),
            logger=logger,
        )
        rows.append({
            "model": args.model, "dataset": bundle.source,
            "compensator": display_name, "rank": comp_rank,
            "calib_size": args.calib_size, "epochs": args.epochs,
            "final_calib_loss": round(final_loss, 6),
            **metrics,
        })
        logger.info("")

    # ── 4. 写入结果 ───────────────────────────────────────────────────────
    saved = write_csv(csv_path, rows)
    logger.info(f"结果已保存至：{saved}")

    # ── 5. 打印摘要表格 ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("摘要（按 Top-1 降序）")
    logger.info("=" * 60)
    header = f"{'方法':<18} {'Top-1':>7} {'延迟(ms)':>10} {'内存(MB)':>10} {'参数量':>12}"
    logger.info(header)
    logger.info("-" * 60)
    for row in sorted(rows, key=lambda r: r["top1"], reverse=True):
        logger.info(
            f"{row['compensator']:<18} "
            f"{row['top1']:>6.2f}% "
            f"{row['latency_ms']:>10.3f} "
            f"{row['peak_memory_mb']:>10.2f} "
            f"{row['params']:>12,}"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()