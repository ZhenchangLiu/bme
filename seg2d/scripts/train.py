#!/usr/bin/env python3
"""Train the hand-written U-Net baseline on Fundus-AVSeg."""

from __future__ import annotations

import argparse
import csv
import os
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
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
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
    from torch.nn.parallel import DistributedDataParallel

    from seg2d.losses import CrossEntropyDiceLoss
    from seg2d.models import build_unet

    dist_info = setup_distributed(device_arg)
    seed = int(config["project"].get("seed", 42))
    set_seed(seed)
    device = dist_info["device"]

    train_loader, val_loader, train_sampler = build_dataloaders(config, distributed=dist_info["enabled"])
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
    scheduler = build_scheduler(config, optimizer)
    amp_enabled = bool(train_config.get("amp", False)) and device.type == "cuda"
    scaler = make_grad_scaler(device, amp_enabled)

    start_epoch = 1
    best_vessel_dice = -1.0
    if resume_path is not None:
        checkpoint = load_checkpoint(resolve_path(resume_path), device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_vessel_dice = float(checkpoint.get("best_vessel_dice", best_vessel_dice))

    if dist_info["enabled"]:
        device_ids = [dist_info["local_rank"]] if device.type == "cuda" else None
        model = DistributedDataParallel(model, device_ids=device_ids)

    output_config = config["output"]
    output_dir = resolve_path(output_config["root"])
    checkpoint_dir = resolve_path(output_config["checkpoint_dir"])
    if is_main_process(dist_info):
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(dist_info)
    history_path = output_dir / "history.csv"

    epochs = int(train_config["epochs"])
    last_metrics: dict[str, Any] = {}
    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=int(model_config.get("num_classes", 5)),
            ignore_index=int(data_config["ignore_index"]),
            scaler=scaler,
            amp_enabled=amp_enabled,
            dist_info=dist_info,
            max_batches=limit_train_batches,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=int(model_config.get("num_classes", 5)),
            ignore_index=int(data_config["ignore_index"]),
            amp_enabled=amp_enabled,
            dist_info=dist_info,
            max_batches=limit_val_batches,
        )
        step_scheduler(scheduler, val_metrics)
        learning_rate = optimizer.param_groups[0]["lr"]

        is_best = val_metrics["vessel_dice"] > best_vessel_dice
        if is_best:
            best_vessel_dice = val_metrics["vessel_dice"]

        checkpoint = {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if amp_enabled else None,
            "config": config,
            "best_vessel_dice": best_vessel_dice,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        if is_main_process(dist_info):
            save_checkpoint(checkpoint, checkpoint_dir / "last.pt")
            if is_best:
                save_checkpoint(checkpoint, checkpoint_dir / "best.pt")

            maybe_save_val_predictions(
                model=unwrap_model(model),
                loader=val_loader,
                device=device,
                output_dir=output_dir,
                epoch=epoch,
                config=config,
                amp_enabled=amp_enabled,
            )

            row = flatten_history(epoch, train_metrics, val_metrics, best_vessel_dice, learning_rate)
            append_history(history_path, row)
            print(format_epoch(epoch, epochs, train_metrics, val_metrics, best_vessel_dice, learning_rate))
        distributed_barrier(dist_info)
        last_metrics = {"train": train_metrics, "val": val_metrics, "best_vessel_dice": best_vessel_dice}

    result = {
        "model": model,
        "optimizer": optimizer,
        "last_metrics": last_metrics,
        "checkpoint_dir": checkpoint_dir,
        "output_dir": output_dir,
        "dist": dist_info,
    }
    distributed_barrier(dist_info)
    cleanup_distributed(dist_info)
    return result


def build_dataloaders(config: dict[str, Any], distributed: bool = False):
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from seg2d.datasets import FundusAVSegDataset
    from seg2d.datasets.transforms import build_train_transform

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
        transform=build_train_transform(config),
    )
    val_dataset = FundusAVSegDataset(
        root=root,
        split="val",
        image_size=image_size,
        val_fraction=val_fraction,
        seed=seed,
        return_meta=False,
    )
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True, seed=seed, drop_last=False)
        if distributed
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
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
    return train_loader, val_loader, train_sampler


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    num_classes: int,
    ignore_index: int | None,
    scaler,
    amp_enabled: bool,
    dist_info: dict[str, Any] | None = None,
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
        with autocast_context(device, amp_enabled):
            logits = model(images)
            losses = criterion(logits, masks)
        if amp_enabled:
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            optimizer.step()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += float(losses["loss"].detach().item()) * batch_size
        total_ce += float(losses["ce_loss"].detach().item()) * batch_size
        total_dice_loss += float(losses["dice_loss"].detach().item()) * batch_size

        with torch.no_grad():
            update_confusions(logits, masks, confusion, vessel_confusion, num_classes, ignore_index)

    total_loss, total_ce, total_dice_loss, total_samples = reduce_epoch_state(
        total_loss=total_loss,
        total_ce=total_ce,
        total_dice_loss=total_dice_loss,
        total_samples=total_samples,
        confusion=confusion,
        vessel_confusion=vessel_confusion,
        device=device,
        dist_info=dist_info,
    )
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
    amp_enabled: bool = False,
    dist_info: dict[str, Any] | None = None,
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
            with autocast_context(device, amp_enabled):
                logits = model(images)
                losses = criterion(logits, masks)

            batch_size = images.size(0)
            total_samples += batch_size
            total_loss += float(losses["loss"].detach().item()) * batch_size
            total_ce += float(losses["ce_loss"].detach().item()) * batch_size
            total_dice_loss += float(losses["dice_loss"].detach().item()) * batch_size

            update_confusions(logits, masks, confusion, vessel_confusion, num_classes, ignore_index)

    total_loss, total_ce, total_dice_loss, total_samples = reduce_epoch_state(
        total_loss=total_loss,
        total_ce=total_ce,
        total_dice_loss=total_dice_loss,
        total_samples=total_samples,
        confusion=confusion,
        vessel_confusion=vessel_confusion,
        device=device,
        dist_info=dist_info,
    )
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


def reduce_epoch_state(
    total_loss: float,
    total_ce: float,
    total_dice_loss: float,
    total_samples: int,
    confusion: dict[str, Any],
    vessel_confusion: dict[str, Any],
    device,
    dist_info: dict[str, Any] | None,
) -> tuple[float, float, float, int]:
    if not dist_info or not dist_info.get("enabled", False):
        return total_loss, total_ce, total_dice_loss, total_samples

    import torch
    import torch.distributed as dist

    totals = torch.tensor(
        [total_loss, total_ce, total_dice_loss, float(total_samples)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    for bucket in (confusion, vessel_confusion):
        for value in bucket.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)

    return (
        float(totals[0].item()),
        float(totals[1].item()),
        float(totals[2].item()),
        int(totals[3].item()),
    )


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


def build_scheduler(config: dict[str, Any], optimizer):
    import torch

    scheduler_config = config["train"].get("scheduler", {})
    if not scheduler_config or not scheduler_config.get("enabled", False):
        return None

    name = str(scheduler_config.get("name", "cosine")).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_config.get("t_max", config["train"]["epochs"])),
            eta_min=float(scheduler_config.get("min_lr", 0.0)),
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 5)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def step_scheduler(scheduler, val_metrics: dict[str, Any]) -> None:
    if scheduler is None:
        return
    if scheduler.__class__.__name__ == "ReduceLROnPlateau":
        scheduler.step(val_metrics["vessel_dice"])
    else:
        scheduler.step()


def maybe_save_val_predictions(
    model,
    loader,
    device,
    output_dir: Path,
    epoch: int,
    config: dict[str, Any],
    amp_enabled: bool,
) -> None:
    logging_config = config.get("logging", {})
    if not logging_config.get("save_val_predictions", False):
        return

    interval = int(logging_config.get("prediction_interval", 1))
    if interval <= 0 or epoch % interval != 0:
        return

    import numpy as np
    import torch
    from PIL import Image

    from seg2d.utils.mask import class_to_rgb

    num_samples = int(logging_config.get("num_val_predictions", 4))
    save_dir = output_dir / "val_predictions" / f"epoch_{epoch:04d}"
    save_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with autocast_context(device, amp_enabled):
                logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            for index in range(images.size(0)):
                if saved >= num_samples:
                    break
                image = tensor_image_to_pil(images[index].detach().cpu())
                target = Image.fromarray(class_to_rgb(masks[index].detach().cpu().numpy().astype(np.uint8)))
                pred = Image.fromarray(class_to_rgb(predictions[index].detach().cpu().numpy().astype(np.uint8)))
                overlay = Image.blend(image, pred, alpha=float(logging_config.get("overlay_alpha", 0.45)))

                stem = f"sample_{saved:02d}"
                image.save(save_dir / f"{stem}_image.png")
                target.save(save_dir / f"{stem}_target.png")
                pred.save(save_dir / f"{stem}_pred.png")
                overlay.save(save_dir / f"{stem}_overlay.png")
                saved += 1
            if saved >= num_samples:
                break

    if was_training:
        model.train()


def tensor_image_to_pil(image) -> "Image.Image":
    import numpy as np
    from PIL import Image

    array = image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required for training: pip install pyyaml") from exc

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_checkpoint(path: Path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


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
    if args.no_amp:
        config["train"]["amp"] = False


def make_device(device_arg: str | None):
    import torch

    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_distributed(device_arg: str | None) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return {
            "enabled": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": make_device(device_arg),
            "backend": None,
        }

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = make_device(device_arg)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return {
        "enabled": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "backend": backend,
    }


def is_main_process(dist_info: dict[str, Any] | None) -> bool:
    return dist_info is None or int(dist_info.get("rank", 0)) == 0


def distributed_barrier(dist_info: dict[str, Any] | None) -> None:
    if not dist_info or not dist_info.get("enabled", False):
        return

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed(dist_info: dict[str, Any] | None) -> None:
    if not dist_info or not dist_info.get("enabled", False):
        return

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def make_grad_scaler(device, enabled: bool):
    import torch

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device, enabled: bool):
    import torch

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


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
    learning_rate: float,
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
        "learning_rate": learning_rate,
    }
    for class_id, score in train_metrics["per_class_dice"].items():
        row[f"train_dice_class_{class_id}"] = score
    for class_id, score in val_metrics["per_class_dice"].items():
        row[f"val_dice_class_{class_id}"] = score
    for class_id, score in train_metrics["per_class_iou"].items():
        row[f"train_iou_class_{class_id}"] = score
    for class_id, score in val_metrics["per_class_iou"].items():
        row[f"val_iou_class_{class_id}"] = score
    return row


def format_epoch(
    epoch: int,
    epochs: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    best_vessel_dice: float,
    learning_rate: float,
) -> str:
    return (
        f"epoch {epoch}/{epochs} "
        f"train_loss={train_metrics['loss']:.6f} "
        f"val_loss={val_metrics['loss']:.6f} "
        f"val_vessel_dice={val_metrics['vessel_dice']:.6f} "
        f"val_vessel_iou={val_metrics['vessel_iou']:.6f} "
        f"best_vessel_dice={best_vessel_dice:.6f} "
        f"lr={learning_rate:.6g}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
