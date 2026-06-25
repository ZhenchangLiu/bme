#!/usr/bin/env python3
"""Train the hand-written U-Net baseline on Fundus-AVSeg."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Fundus-AVSeg U-Net.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "fundus_avseg_unet.yaml",
        help="Training config path.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override config train.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config train.batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override config train.num_workers.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override config train.learning_rate.")
    parser.add_argument("--device", default=None, help="Device string, for example cuda, cuda:0, or cpu.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume checkpoint path.")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="Debug limit for train batches.")
    parser.add_argument("--limit-val-batches", type=int, default=None, help="Debug limit for val batches.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    run_training(
        config=config,
        device_arg=args.device,
        resume_path=args.resume,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )
    return 0


def run_training(
    config: dict[str, Any],
    device_arg: str | None = None,
    resume_path: Path | None = None,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
) -> dict[str, Any]:
    import torch

    from seg2d.losses import CrossEntropyDiceLoss
    from seg2d.models import build_unet

    seed = int(config["project"].get("seed", 42))
    set_seed(seed)
    device = make_device(device_arg)

    train_loader, val_loader = build_dataloaders(config)
    model_config = config["model"]
    model = build_unet(
        in_channels=int(model_config.get("in_channels", 3)),
        num_classes=int(model_config.get("num_classes", 5)),
        base_channels=int(model_config.get("base_channels", 64)),
        bilinear=bool(model_config.get("bilinear", True)),
    ).to(device)

    train_config = config["train"]
    data_config = config["data"]
    loss_config = train_config.get("loss", {})
    criterion = CrossEntropyDiceLoss(
        num_classes=int(model_config.get("num_classes", 5)),
        ignore_index=int(data_config["ignore_index"]),
        ce_weight=float(loss_config.get("ce_weight", 1.0)),
        dice_weight=float(loss_config.get("dice_weight", 1.0)),
        include_background=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )

    start_epoch = 1
    best_vessel_dice = -1.0
    if resume_path is not None:
        checkpoint = torch.load(resolve_path(resume_path), map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_vessel_dice = float(checkpoint.get("best_vessel_dice", best_vessel_dice))

    output_config = config["output"]
    output_dir = resolve_path(output_config["root"])
    checkpoint_dir = resolve_path(output_config["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"

    epochs = int(train_config["epochs"])
    last_metrics: dict[str, Any] = {}
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=int(model_config.get("num_classes", 5)),
            ignore_index=int(data_config["ignore_index"]),
            max_batches=limit_train_batches,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=int(model_config.get("num_classes", 5)),
            ignore_index=int(data_config["ignore_index"]),
            max_batches=limit_val_batches,
        )

        is_best = val_metrics["vessel_dice"] > best_vessel_dice
        if is_best:
            best_vessel_dice = val_metrics["vessel_dice"]

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "best_vessel_dice": best_vessel_dice,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        save_checkpoint(checkpoint, checkpoint_dir / "last.pt")
        if is_best:
            save_checkpoint(checkpoint, checkpoint_dir / "best.pt")

        row = flatten_history(epoch, train_metrics, val_metrics, best_vessel_dice)
        append_history(history_path, row)
        print(format_epoch(epoch, epochs, train_metrics, val_metrics, best_vessel_dice))
        last_metrics = {"train": train_metrics, "val": val_metrics, "best_vessel_dice": best_vessel_dice}

    return {
        "model": model,
        "optimizer": optimizer,
        "last_metrics": last_metrics,
        "checkpoint_dir": checkpoint_dir,
        "output_dir": output_dir,
    }


def build_dataloaders(config: dict[str, Any]):
    import torch
    from torch.utils.data import DataLoader

    from seg2d.datasets import FundusAVSegDataset

    data_config = config["data"]
    train_config = config["train"]
    root = resolve_path(data_config["root"])
    image_size = tuple(data_config["image_size"])
    val_fraction = float(data_config.get("val_fraction", 0.1))
    seed = int(config["project"].get("seed", 42))
    batch_size = int(train_config["batch_size"])
    num_workers = int(train_config.get("num_workers", 0))

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataset = FundusAVSegDataset(
        root=root,
        split="train",
        image_size=image_size,
        val_fraction=val_fraction,
        seed=seed,
        return_meta=False,
    )
    val_dataset = FundusAVSegDataset(
        root=root,
        split="val",
        image_size=image_size,
        val_fraction=val_fraction,
        seed=seed,
        return_meta=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    num_classes: int,
    ignore_index: int | None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    import torch

    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_dice_loss = 0.0
    total_samples = 0
    confusion = make_confusion(num_classes, device)
    vessel_confusion = make_binary_confusion(device)

    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        losses = criterion(logits, masks)
        losses["loss"].backward()
        optimizer.step()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += float(losses["loss"].detach().item()) * batch_size
        total_ce += float(losses["ce_loss"].detach().item()) * batch_size
        total_dice_loss += float(losses["dice_loss"].detach().item()) * batch_size

        with torch.no_grad():
            update_confusions(logits, masks, confusion, vessel_confusion, num_classes, ignore_index)

    return summarize_epoch(
        total_loss,
        total_ce,
        total_dice_loss,
        total_samples,
        confusion,
        vessel_confusion,
        num_classes,
        ignore_index,
    )


def evaluate(
    model,
    loader,
    criterion,
    device,
    num_classes: int,
    ignore_index: int | None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    import torch

    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_dice_loss = 0.0
    total_samples = 0
    confusion = make_confusion(num_classes, device)
    vessel_confusion = make_binary_confusion(device)

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            losses = criterion(logits, masks)

            batch_size = images.size(0)
            total_samples += batch_size
            total_loss += float(losses["loss"].detach().item()) * batch_size
            total_ce += float(losses["ce_loss"].detach().item()) * batch_size
            total_dice_loss += float(losses["dice_loss"].detach().item()) * batch_size

            update_confusions(logits, masks, confusion, vessel_confusion, num_classes, ignore_index)

    return summarize_epoch(
        total_loss,
        total_ce,
        total_dice_loss,
        total_samples,
        confusion,
        vessel_confusion,
        num_classes,
        ignore_index,
    )


def update_confusions(
    logits,
    masks,
    confusion: dict[str, Any],
    vessel_confusion: dict[str, Any],
    num_classes: int,
    ignore_index: int | None,
) -> None:
    from seg2d.metrics import logits_to_prediction, per_class_confusion

    prediction = logits_to_prediction(logits)
    batch_confusion = per_class_confusion(prediction, masks, num_classes, ignore_index)
    for key in confusion:
        confusion[key] += batch_confusion[key]

    batch_vessel = binary_confusion(prediction, masks, positive_class_ids=(1, 2, 3), ignore_index=ignore_index)
    for key in vessel_confusion:
        vessel_confusion[key] += batch_vessel[key]


def summarize_epoch(
    total_loss: float,
    total_ce: float,
    total_dice_loss: float,
    total_samples: int,
    confusion: dict[str, Any],
    vessel_confusion: dict[str, Any],
    num_classes: int,
    ignore_index: int | None,
) -> dict[str, Any]:
    from seg2d.metrics import dice_from_confusion, iou_from_confusion

    if total_samples == 0:
        raise RuntimeError("No samples were processed.")

    class_ids = tuple(
        class_id
        for class_id in range(num_classes)
        if ignore_index is None or class_id != ignore_index
    )
    per_class_dice = dice_from_confusion(confusion, class_ids)
    per_class_iou = iou_from_confusion(confusion, class_ids)
    vessel_dice, vessel_iou = binary_scores(vessel_confusion)
    return {
        "loss": total_loss / total_samples,
        "ce_loss": total_ce / total_samples,
        "dice_loss": total_dice_loss / total_samples,
        "per_class_dice": per_class_dice,
        "per_class_iou": per_class_iou,
        "vessel_dice": vessel_dice,
        "vessel_iou": vessel_iou,
        "samples": total_samples,
    }


def make_confusion(num_classes: int, device):
    import torch

    return {
        "tp": torch.zeros(num_classes, dtype=torch.float64, device=device),
        "fp": torch.zeros(num_classes, dtype=torch.float64, device=device),
        "fn": torch.zeros(num_classes, dtype=torch.float64, device=device),
    }


def make_binary_confusion(device):
    import torch

    return {
        "tp": torch.zeros((), dtype=torch.float64, device=device),
        "fp": torch.zeros((), dtype=torch.float64, device=device),
        "fn": torch.zeros((), dtype=torch.float64, device=device),
    }


def binary_confusion(prediction, target, positive_class_ids=(1, 2, 3), ignore_index: int | None = None):
    import torch

    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid = target != ignore_index
    positives = torch.tensor(tuple(positive_class_ids), device=prediction.device)
    pred_pos = torch.isin(prediction, positives) & valid
    target_pos = torch.isin(target, positives) & valid
    return {
        "tp": torch.sum(pred_pos & target_pos, dtype=torch.float64),
        "fp": torch.sum(pred_pos & ~target_pos & valid, dtype=torch.float64),
        "fn": torch.sum(~pred_pos & target_pos & valid, dtype=torch.float64),
    }


def binary_scores(confusion: dict[str, Any], eps: float = 1e-7) -> tuple[float, float]:
    tp, fp, fn = confusion["tp"], confusion["fp"], confusion["fn"]
    dice = ((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)).item()
    iou = ((tp + eps) / (tp + fp + fn + eps)).item()
    return dice, iou


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required for training: pip install pyyaml") from exc

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers
    if args.learning_rate is not None:
        config["train"]["learning_rate"] = args.learning_rate
    if args.limit_train_batches is not None:
        config["train"]["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        config["train"]["limit_val_batches"] = args.limit_val_batches


def make_device(device_arg: str | None):
    import torch

    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def save_checkpoint(checkpoint: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def flatten_history(
    epoch: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    best_vessel_dice: float,
) -> dict[str, Any]:
    row = {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_ce_loss": train_metrics["ce_loss"],
        "train_dice_loss": train_metrics["dice_loss"],
        "train_vessel_dice": train_metrics["vessel_dice"],
        "train_vessel_iou": train_metrics["vessel_iou"],
        "val_loss": val_metrics["loss"],
        "val_ce_loss": val_metrics["ce_loss"],
        "val_dice_loss": val_metrics["dice_loss"],
        "val_vessel_dice": val_metrics["vessel_dice"],
        "val_vessel_iou": val_metrics["vessel_iou"],
        "best_vessel_dice": best_vessel_dice,
    }
    for class_id, score in val_metrics["per_class_dice"].items():
        row[f"val_dice_class_{class_id}"] = score
    return row


def format_epoch(
    epoch: int,
    epochs: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    best_vessel_dice: float,
) -> str:
    return (
        f"epoch {epoch}/{epochs} "
        f"train_loss={train_metrics['loss']:.6f} "
        f"val_loss={val_metrics['loss']:.6f} "
        f"val_vessel_dice={val_metrics['vessel_dice']:.6f} "
        f"val_vessel_iou={val_metrics['vessel_iou']:.6f} "
        f"best_vessel_dice={best_vessel_dice:.6f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
