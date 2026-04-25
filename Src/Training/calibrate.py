"""Src/Models_Training/calibrate.py"""
import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from Src.Training.trainer import feature_mse_loss, logit_mse_loss


def calibrate_compensators(
    model: torch.nn.Module,
    calibration_loader,
    device: torch.device,
    removed_blocks: list[str] | None = None,
    epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    feature_loss_weight: float = 1.0,
    logit_loss_weight: float = 1.0,
    grad_clip: float | None = 1.0,
    max_batches: int | None = None,
) -> dict[str, list[float]]:
    if not hasattr(model, "freeze_backbone") or not hasattr(model, "compensator_parameters"):
        raise AttributeError("Model must expose freeze_backbone() and compensator_parameters().")

    removed_blocks = removed_blocks or list(model.get_block_names())
    model.to(device)
    model.freeze_backbone()
    params = model.compensator_parameters()
    if not params:
        return {"epoch_loss": [0.0]}

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    history: list[float] = []

    model.eval()
    for _ in range(epochs):
        total_loss = 0.0
        total_steps = 0
        for batch_index, (images, _) in enumerate(calibration_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device)
            with torch.no_grad():
                teacher = model(images, mode="full", return_features=True)
            student = model(
                images,
                mode="compensated",
                removed_blocks=removed_blocks,
                return_features=True,
            )

            loss = images.new_zeros(())
            if feature_loss_weight > 0:
                loss = loss + feature_loss_weight * feature_mse_loss(
                    student["features"],
                    teacher["features"],
                    layers=removed_blocks,
                )
            if logit_loss_weight > 0:
                loss = loss + logit_loss_weight * logit_mse_loss(student["logits"], teacher["logits"])

            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()

            total_loss += float(loss.item())
            total_steps += 1

        history.append(total_loss / max(total_steps, 1))

    return {"epoch_loss": history}
