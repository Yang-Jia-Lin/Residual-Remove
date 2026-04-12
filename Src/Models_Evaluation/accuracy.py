# src/evaluation/accuracy.py
"""Top-K 准确率计算
注意：topk_accuracy 返回 0~100 的百分比
"""
from __future__ import annotations
from collections.abc import Iterable
import torch


def extract_logits(output: torch.Tensor | dict) -> torch.Tensor:
    """从模型输出中提取 logits 张量。
    
    InjectedModel 在 return_features=True 时返回 dict，
    普通前向返回 Tensor，这个函数统一处理两种情况。
    """
    if isinstance(output, dict):
        return output["logits"]
    return output


def topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Iterable[int] = (1, 5),
) -> list[torch.Tensor]:
    """计算 Top-K 准确率，返回百分比（0~100）。
    
    Args:
        logits:  模型输出的原始分数，shape (B, num_classes)
        targets: 真实标签，shape (B,)
        topk:    需要计算的 K 值列表，默认 (1, 5)
    
    Returns:
        与 topk 等长的列表，每个元素是对应 K 值下的准确率百分比。
    """
    topk = list(topk)
    max_k = max(topk)
    batch_size = targets.size(0)

    # topk 返回 (values, indices)，我们只要 indices
    _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
    # 转置为 (max_k, B)，方便逐行比对
    predictions = predictions.t()
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))

    results: list[torch.Tensor] = []
    for k in topk:
        # 前 k 行里只要有一行猜对了，这个样本就算正确
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append(correct_k.mul_(100.0 / batch_size))
    return results