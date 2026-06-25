#!/usr/bin/env python3
"""Evaluate a trained U-Net checkpoint on a Fundus-AVSeg split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train import evaluate as evaluate_loop
from train import load_config, make_device, resolve_path


DEFAULT_CONFIG = PROJECT_DIR / "configs" / "fundus_avseg_unet.yaml"
DEFAULT_CHECKPOINT = PROJECT_DIR / "checkpoints" / "fundus_avseg_unet" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Fundus-AVSeg U-Net checkpoint.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Fallback config path.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Checkpoint path.")
    parser.add_argument("--split", default="test", choices=["train", "val", "trainval", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument("--device", default=None, help="Device string, for example cuda, cuda:0, or cpu.")
    parser.add_argument("--limit-batches", type=int, default=None, help="Debug limit for eval batches.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch

    from seg2d.losses import CrossEntropyDiceLoss
    from seg2d.models import build_unet

    device = make_device(args.device)
    checkpoint = load_checkpoint(resolve_path(args.checkpoint), device)
    config = checkpoint.get("config") or load_config(args.config)
    apply_eval_overrides(config, args)

    model_config = config["model"]
    data_config = config["data"]
    train_config = config["train"]
    loss_config = train_config.get("loss", {})

    model = build_unet(
        in_channels=int(model_config.get("in_channels", 3)),
        num_classes=int(model_config.get("num_classes", 5)),
        base_channels=int(model_config.get("base_channels", 64)),
        bilinear=bool(model_config.get("bilinear", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    criterion = CrossEntropyDiceLoss(
        num_classes=int(model_config.get("num_classes", 5)),
        ignore_index=int(data_config["ignore_index"]),
        ce_weight=float(loss_config.get("ce_weight", 1.0)),
        dice_weight=float(loss_config.get("dice_weight", 1.0)),
        include_background=True,
    )
    loader = build_eval_loader(config, split=args.split)
    metrics = evaluate_loop(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        num_classes=int(model_config.get("num_classes", 5)),
        ignore_index=int(data_config["ignore_index"]),
        max_batches=args.limit_batches,
    )

    result = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "split": args.split,
        "device": str(device),
        "epoch": checkpoint.get("epoch"),
        "metrics": to_jsonable(metrics),
    }
    print_summary(result)

    if args.output_json is not None:
        output_path = resolve_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved: {output_path}")
    return 0


def build_eval_loader(config: dict[str, Any], split: str):
    import torch
    from torch.utils.data import DataLoader

    from seg2d.datasets import FundusAVSegDataset

    data_config = config["data"]
    train_config = config["train"]
    dataset = FundusAVSegDataset(
        root=resolve_path(data_config["root"]),
        split=split,
        image_size=tuple(data_config["image_size"]),
        val_fraction=float(data_config.get("val_fraction", 0.1)),
        seed=int(config["project"].get("seed", 42)),
        return_meta=False,
    )
    return DataLoader(
        dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        num_workers=int(train_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def apply_eval_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers


def load_checkpoint(path: Path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def print_summary(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    print(f"checkpoint: {result['checkpoint']}")
    print(f"split: {result['split']}")
    print(f"device: {result['device']}")
    print(f"epoch: {result['epoch']}")
    print(f"loss: {metrics['loss']:.6f}")
    print(f"ce_loss: {metrics['ce_loss']:.6f}")
    print(f"dice_loss: {metrics['dice_loss']:.6f}")
    print(f"vessel_dice: {metrics['vessel_dice']:.6f}")
    print(f"vessel_iou: {metrics['vessel_iou']:.6f}")
    print(f"per_class_dice: {format_scores(metrics['per_class_dice'])}")
    print(f"per_class_iou: {format_scores(metrics['per_class_iou'])}")


def format_scores(scores: dict[str, float]) -> str:
    return ", ".join(f"{class_id}={score:.6f}" for class_id, score in scores.items())


if __name__ == "__main__":
    raise SystemExit(main())
