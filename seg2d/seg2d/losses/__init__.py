"""Segmentation loss functions."""

from seg2d.losses.combined import CrossEntropyDiceLoss
from seg2d.losses.dice import DiceLoss

__all__ = ["CrossEntropyDiceLoss", "DiceLoss"]
