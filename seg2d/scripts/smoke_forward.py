#!/usr/bin/env python3
"""Run a one-batch Fundus-AVSeg U-Net forward/backward smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Fundus-AVSeg + U-Net.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_DIR / "data" / "Fundus-AVSeg",
        help="Path to the extracted Fundus-AVSeg dataset.",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "trainval", "test", "all"])
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--ignore-index", type=int, default=4)
    parser.add_argument("--transpose-conv", action="store_true", help="Use transposed convolutions for upsampling.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from seg2d.datasets import FundusAVSegDataset
    from seg2d.models import build_unet

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = FundusAVSegDataset(
        root=args.data_root,
        split=args.split,
        image_size=tuple(args.image_size),
        return_meta=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    batch = next(iter(loader))
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)

    model = build_unet(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        bilinear=not args.transpose_conv,
    ).to(device)
    model.train()

    logits = model(images)
    loss = F.cross_entropy(logits, masks, ignore_index=args.ignore_index)
    loss.backward()

    grad_norm = model.outc.conv.weight.grad.detach().norm().item()

    print(f"device: {device}")
    print(f"dataset length: {len(dataset)}")
    print(f"image shape: {tuple(images.shape)} {images.dtype}")
    print(f"mask shape: {tuple(masks.shape)} {masks.dtype}")
    print(f"logits shape: {tuple(logits.shape)} {logits.dtype}")
    print(f"loss: {loss.item():.6f}")
    print(f"output grad norm: {grad_norm:.6f}")
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
