"""Segmentation metrics for class-id masks."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def logits_to_prediction(logits: torch.Tensor) -> torch.Tensor:
    """Convert BxCxHxW logits to BxHxW class predictions."""
    if logits.ndim != 4:
        raise ValueError(f"Expected logits shape BxCxHxW, got {tuple(logits.shape)}")
    return torch.argmax(logits, dim=1)


def per_class_confusion(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compute TP/FP/FN for each class."""
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    if prediction.ndim != 3:
        raise ValueError(f"Expected BxHxW prediction and target, got {prediction.shape}")

    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid = target != ignore_index

    prediction = prediction[valid]
    target = target[valid]

    true_positive = torch.zeros(num_classes, dtype=torch.float64, device=prediction.device)
    false_positive = torch.zeros_like(true_positive)
    false_negative = torch.zeros_like(true_positive)

    for class_id in range(num_classes):
        pred_class = prediction == class_id
        target_class = target == class_id
        true_positive[class_id] = torch.sum(pred_class & target_class)
        false_positive[class_id] = torch.sum(pred_class & ~target_class)
        false_negative[class_id] = torch.sum(~pred_class & target_class)

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }


def dice_from_confusion(
    confusion: dict[str, torch.Tensor],
    class_ids: Iterable[int],
    eps: float = 1e-7,
) -> dict[int, float]:
    """Return Dice scores for selected classes."""
    tp, fp, fn = confusion["tp"], confusion["fp"], confusion["fn"]
    scores: dict[int, float] = {}
    for class_id in class_ids:
        numerator = 2.0 * tp[class_id]
        denominator = 2.0 * tp[class_id] + fp[class_id] + fn[class_id]
        scores[class_id] = ((numerator + eps) / (denominator + eps)).item()
    return scores


def iou_from_confusion(
    confusion: dict[str, torch.Tensor],
    class_ids: Iterable[int],
    eps: float = 1e-7,
) -> dict[int, float]:
    """Return IoU scores for selected classes."""
    tp, fp, fn = confusion["tp"], confusion["fp"], confusion["fn"]
    scores: dict[int, float] = {}
    for class_id in class_ids:
        denominator = tp[class_id] + fp[class_id] + fn[class_id]
        scores[class_id] = ((tp[class_id] + eps) / (denominator + eps)).item()
    return scores


def binary_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    positive_class_ids: Iterable[int],
    ignore_index: int | None = None,
    eps: float = 1e-7,
) -> float:
    """Compute Dice for merged positive classes."""
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")

    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid = target != ignore_index

    positives = torch.tensor(tuple(positive_class_ids), device=prediction.device)
    pred_pos = torch.isin(prediction, positives) & valid
    target_pos = torch.isin(target, positives) & valid

    intersection = torch.sum(pred_pos & target_pos, dtype=torch.float64)
    denominator = torch.sum(pred_pos, dtype=torch.float64) + torch.sum(target_pos, dtype=torch.float64)
    return ((2.0 * intersection + eps) / (denominator + eps)).item()


def binary_iou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    positive_class_ids: Iterable[int],
    ignore_index: int | None = None,
    eps: float = 1e-7,
) -> float:
    """Compute IoU for merged positive classes."""
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")

    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid = target != ignore_index

    positives = torch.tensor(tuple(positive_class_ids), device=prediction.device)
    pred_pos = torch.isin(prediction, positives) & valid
    target_pos = torch.isin(target, positives) & valid

    intersection = torch.sum(pred_pos & target_pos, dtype=torch.float64)
    union = torch.sum(pred_pos | target_pos, dtype=torch.float64)
    return ((intersection + eps) / (union + eps)).item()


def segmentation_metrics(
    logits_or_prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
    class_ids: Iterable[int] | None = None,
    vessel_class_ids: Iterable[int] = (1, 2, 3),
) -> dict[str, object]:
    """Compute per-class and merged-vessel Dice/IoU metrics."""
    if logits_or_prediction.ndim == 4:
        prediction = logits_to_prediction(logits_or_prediction)
    else:
        prediction = logits_or_prediction

    if class_ids is None:
        class_ids = tuple(
            class_id
            for class_id in range(num_classes)
            if ignore_index is None or class_id != ignore_index
        )
    else:
        class_ids = tuple(class_ids)

    confusion = per_class_confusion(
        prediction=prediction,
        target=target,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    return {
        "per_class_dice": dice_from_confusion(confusion, class_ids),
        "per_class_iou": iou_from_confusion(confusion, class_ids),
        "vessel_dice": binary_dice(prediction, target, vessel_class_ids, ignore_index),
        "vessel_iou": binary_iou(prediction, target, vessel_class_ids, ignore_index),
    }
