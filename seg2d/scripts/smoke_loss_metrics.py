#!/usr/bin/env python3
"""Run a one-batch loss/metrics smoke test on Fundus-AVSeg."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test loss and metrics on Fundus-AVSeg.")
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
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--ignore-index", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    from seg2d.datasets import FundusAVSegDataset
    from seg2d.losses import CrossEntropyDiceLoss
    from seg2d.metrics import segmentation_metrics
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

    model = build_unet(num_classes=args.num_classes, base_channels=args.base_channels).to(device)
    model.train()

    logits = model(images)
    criterion = CrossEntropyDiceLoss(
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        ce_weight=1.0,
        dice_weight=1.0,
        include_background=True,
    )
    losses = criterion(logits, masks)
    losses["loss"].backward()

    metrics = segmentation_metrics(
        logits,
        masks,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        vessel_class_ids=(1, 2, 3),
    )

    print(f"device: {device}")
    print(f"dataset length: {len(dataset)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"loss: {losses['loss'].item():.6f}")
    print(f"ce_loss: {losses['ce_loss'].item():.6f}")
    print(f"dice_loss: {losses['dice_loss'].item():.6f}")
    print(f"per_class_dice: {format_scores(metrics['per_class_dice'])}")
    print(f"per_class_iou: {format_scores(metrics['per_class_iou'])}")
    print(f"vessel_dice: {metrics['vessel_dice']:.6f}")
    print(f"vessel_iou: {metrics['vessel_iou']:.6f}")
    print("smoke loss/metrics test passed")
    return 0


def format_scores(scores: dict[int, float]) -> str:
    return ", ".join(f"{class_id}={score:.6f}" for class_id, score in scores.items())


if __name__ == "__main__":
    raise SystemExit(main())
