"""Hand-written 2D U-Net for semantic segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn

from seg2d.models.blocks import DoubleConv, Down, OutConv, Up


class UNet(nn.Module):
    """Standard encoder-decoder U-Net with skip connections."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        base_channels: int = 64,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}")

        factor = 2 if bilinear else 1
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def build_unet(
    in_channels: int = 3,
    num_classes: int = 5,
    base_channels: int = 64,
    bilinear: bool = True,
) -> UNet:
    """Factory used by scripts and future training code."""
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        bilinear=bilinear,
    )
