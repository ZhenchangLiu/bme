#!/usr/bin/env python3
"""Run a tiny train/validation step smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from train import load_config, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test one train batch and one val batch.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "fundus_avseg_unet.yaml",
        help="Training config path.",
    )
    parser.add_argument("--device", default=None, help="Device string, for example cuda, cuda:0, or cpu.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config["project"]["name"] = f"{config['project']['name']}_smoke"
    config["train"]["epochs"] = 1
    config["train"]["batch_size"] = 1
    config["train"]["num_workers"] = 0
    config["output"]["root"] = "seg2d/outputs/smoke_train_step"
    config["output"]["checkpoint_dir"] = "seg2d/checkpoints/smoke_train_step"

    result = run_training(
        config=config,
        device_arg=args.device,
        limit_train_batches=1,
        limit_val_batches=1,
    )
    metrics = result["last_metrics"]
    print(f"checkpoint_dir: {result['checkpoint_dir']}")
    print(f"train_loss: {metrics['train']['loss']:.6f}")
    print(f"val_loss: {metrics['val']['loss']:.6f}")
    print(f"val_vessel_dice: {metrics['val']['vessel_dice']:.6f}")
    print("smoke train step passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
