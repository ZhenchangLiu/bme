"""Dice-based segmentation losses."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss for segmentation logits."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        include_background: bool = True,
        smooth: float = 1.0,
        class_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if smooth < 0:
            raise ValueError(f"smooth must be non-negative, got {smooth}")

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.include_background = include_background
        self.smooth = smooth
        self.class_ids = tuple(class_ids) if class_ids is not None else None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"Expected logits shape BxCxHxW, got {tuple(logits.shape)}")
        if target.ndim != 3:
            raise ValueError(f"Expected target shape BxHxW, got {tuple(target.shape)}")
        if logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
            raise ValueError(
                f"Logits/target shape mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}"
            )
        if logits.shape[1] != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes, got {logits.shape[1]}")

        probabilities = torch.softmax(logits, dim=1)
        valid_mask = torch.ones_like(target, dtype=torch.bool)
        safe_target = target
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            safe_target = target.masked_fill(~valid_mask, 0)

        one_hot = F.one_hot(safe_target.clamp(min=0), num_classes=self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=probabilities.dtype)

        valid_mask = valid_mask.unsqueeze(1)
        probabilities = probabilities * valid_mask
        one_hot = one_hot * valid_mask

        class_ids = self._class_ids(logits.device)
        probabilities = probabilities[:, class_ids]
        one_hot = one_hot[:, class_ids]

        dims = (0, 2, 3)
        intersection = torch.sum(probabilities * one_hot, dim=dims)
        cardinality = torch.sum(probabilities + one_hot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()

    def _class_ids(self, device: torch.device) -> torch.Tensor:
        if self.class_ids is not None:
            class_ids = list(self.class_ids)
        else:
            start = 0 if self.include_background else 1
            class_ids = list(range(start, self.num_classes))

        if self.ignore_index is not None:
            class_ids = [class_id for class_id in class_ids if class_id != self.ignore_index]
        if not class_ids:
            raise ValueError("No classes left for DiceLoss after applying filters.")
        return torch.tensor(class_ids, device=device, dtype=torch.long)
