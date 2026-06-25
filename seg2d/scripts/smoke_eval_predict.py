#!/usr/bin/env python3
"""Run a checkpoint evaluation and prediction smoke test."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "checkpoints" / "fundus_avseg_unet" / "best.pt"
DEFAULT_IMAGE = PROJECT_DIR / "data" / "Fundus-AVSeg" / "images" / "001_G.png"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "smoke_eval_predict"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test checkpoint evaluation and prediction.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None, help="Device string, for example cuda, cuda:0, or cpu.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = resolve_path(args.checkpoint)
    image = resolve_path(args.image)
    output_dir = resolve_path(args.output_dir)

    if not checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")
    if not image.exists():
        raise SystemExit(f"Missing image: {image}")

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_json = output_dir / "eval_val_one_batch.json"
    predictions_dir = output_dir / "predictions"

    evaluate_cmd = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "evaluate.py"),
        "--checkpoint",
        str(checkpoint),
        "--split",
        "val",
        "--batch-size",
        "1",
        "--num-workers",
        "0",
        "--limit-batches",
        "1",
        "--output-json",
        str(eval_json),
    ]
    predict_cmd = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "predict.py"),
        "--checkpoint",
        str(checkpoint),
        "--input",
        str(image),
        "--output-dir",
        str(predictions_dir),
    ]
    if args.device is not None:
        evaluate_cmd.extend(["--device", args.device])
        predict_cmd.extend(["--device", args.device])

    run(evaluate_cmd)
    run(predict_cmd)

    expected_mask = predictions_dir / f"{image.stem}_mask.png"
    expected_overlay = predictions_dir / f"{image.stem}_overlay.png"
    for path in (eval_json, expected_mask, expected_overlay):
        if not path.exists():
            raise SystemExit(f"Expected output was not created: {path}")

    print(f"eval_json: {eval_json}")
    print(f"mask: {expected_mask}")
    print(f"overlay: {expected_overlay}")
    print("smoke eval/predict test passed")
    return 0


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
