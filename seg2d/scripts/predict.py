#!/usr/bin/env python3
"""Run U-Net inference and save visual segmentation outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train import load_config, make_device, resolve_path

from seg2d.utils.mask import class_to_rgb


DEFAULT_CONFIG = PROJECT_DIR / "configs" / "fundus_avseg_unet.yaml"
DEFAULT_CHECKPOINT = PROJECT_DIR / "checkpoints" / "fundus_avseg_unet" / "best.pt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Fundus-AVSeg masks with a U-Net checkpoint.")
    parser.add_argument("--input", type=Path, required=True, help="Input image file or directory.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Checkpoint path.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Fallback config path.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "predictions")
    parser.add_argument("--device", default=None, help="Device string, for example cuda, cuda:0, or cpu.")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay mask opacity.")
    parser.add_argument("--save-npy", action="store_true", help="Save raw class-id masks as .npy files.")
    parser.add_argument(
        "--no-restore-size",
        action="store_true",
        help="Keep outputs at model input size instead of resizing back to original size.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch

    from seg2d.models import build_unet

    device = make_device(args.device)
    checkpoint = load_checkpoint(resolve_path(args.checkpoint), device)
    config = checkpoint.get("config") or load_config(args.config)
    model_config = config["model"]
    image_size = tuple(args.image_size or config["data"]["image_size"])

    model = build_unet(
        in_channels=int(model_config.get("in_channels", 3)),
        num_classes=int(model_config.get("num_classes", 5)),
        base_channels=int(model_config.get("base_channels", 64)),
        bilinear=bool(model_config.get("bilinear", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    input_paths = collect_inputs(resolve_path(args.input))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for image_path in input_paths:
            original_image = Image.open(image_path).convert("RGB")
            tensor = image_to_tensor(original_image, image_size).to(device)
            logits = model(tensor)
            prediction = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

            if not args.no_restore_size:
                prediction_image = Image.fromarray(prediction, mode="L").resize(
                    original_image.size,
                    Image.Resampling.NEAREST,
                )
                prediction = np.asarray(prediction_image, dtype=np.uint8)
                base_image = original_image
            else:
                base_image = original_image.resize(to_pil_size(image_size), Image.Resampling.BILINEAR)

            save_prediction_outputs(
                image_path=image_path,
                output_dir=output_dir,
                base_image=base_image,
                prediction=prediction,
                alpha=args.alpha,
                save_npy=args.save_npy,
            )

    print(f"processed: {len(input_paths)}")
    print(f"output_dir: {output_dir}")
    return 0


def image_to_tensor(image: Image.Image, image_size: tuple[int, int]):
    import torch

    image = image.resize(to_pil_size(image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor


def save_prediction_outputs(
    image_path: Path,
    output_dir: Path,
    base_image: Image.Image,
    prediction: np.ndarray,
    alpha: float,
    save_npy: bool,
) -> None:
    stem = image_path.stem
    rgb_mask = class_to_rgb(prediction)
    mask_image = Image.fromarray(rgb_mask, mode="RGB")
    overlay = Image.blend(base_image.convert("RGB"), mask_image, alpha=alpha)

    mask_path = output_dir / f"{stem}_mask.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    mask_image.save(mask_path)
    overlay.save(overlay_path)
    if save_npy:
        np.save(output_dir / f"{stem}_mask.npy", prediction)
    print(f"saved: {mask_path}")
    print(f"saved: {overlay_path}")


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    paths = sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No images found in directory: {path}")
    return paths


def to_pil_size(image_size: tuple[int, int]) -> tuple[int, int]:
    height, width = image_size
    return width, height


def load_checkpoint(path: Path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


if __name__ == "__main__":
    raise SystemExit(main())
