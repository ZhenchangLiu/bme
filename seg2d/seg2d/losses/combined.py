"""Combined segmentation losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from seg2d.losses.dice import DiceLoss


class CrossEntropyDiceLoss(nn.Module):
    """Weighted sum of cross entropy and soft Dice loss."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        include_background: bool = True,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if ce_weight < 0 or dice_weight < 0:
            raise ValueError("ce_weight and dice_weight must be non-negative.")
        if ce_weight == 0 and dice_weight == 0:
            raise ValueError("At least one loss weight must be positive.")

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            include_background=include_background,
            smooth=smooth,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        ce_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
        dice_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)

        if self.ce_weight > 0:
            if self.ignore_index is None:
                ce_loss = F.cross_entropy(logits, target)
            else:
                ce_loss = F.cross_entropy(logits, target, ignore_index=self.ignore_index)

        if self.dice_weight > 0:
            dice_loss = self.dice(logits, target)

        total = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return {
            "loss": total,
            "ce_loss": ce_loss.detach(),
            "dice_loss": dice_loss.detach(),
        }
