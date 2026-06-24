"""PyTorch dataset adapter for Fundus-AVSeg."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - exercised when torch is absent.
    torch = None
    Dataset = object
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from seg2d.utils.mask import rgb_to_class


SplitName = Literal["train", "val", "trainval", "test", "all"]


@dataclass(frozen=True)
class FundusAVSegSample:
    """A paired Fundus-AVSeg image and annotation path."""

    name: str
    image_path: Path
    mask_path: Path


class FundusAVSegDataset(Dataset):
    """Fundus-AVSeg retinal artery-vein segmentation dataset.

    The dataset returns dictionaries with:

    - `image`: float tensor with shape CxHxW and range [0, 1]
    - `mask`: long tensor with shape HxW
    - `meta`: sample metadata, if `return_meta=True`
    """

    def __init__(
        self,
        root: str | Path,
        split: SplitName = "train",
        image_size: tuple[int, int] | list[int] | None = (512, 512),
        val_fraction: float = 0.1,
        seed: int = 42,
        return_meta: bool = True,
        transform: Callable[[dict], dict] | None = None,
    ) -> None:
        if torch is None:
            raise ImportError(
                "FundusAVSegDataset requires PyTorch. Install torch in the bme conda "
                "environment before training or loading this dataset."
            ) from _TORCH_IMPORT_ERROR

        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.image_size = normalize_image_size(image_size)
        self.val_fraction = val_fraction
        self.seed = seed
        self.return_meta = return_meta
        self.transform = transform
        self.samples = build_samples(
            root=self.root,
            split=split,
            val_fraction=val_fraction,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            original_size = image.size
            if self.image_size is not None:
                image = image.resize(to_pil_size(self.image_size), Image.Resampling.BILINEAR)
            image_array = np.asarray(image, dtype=np.float32) / 255.0

        with Image.open(sample.mask_path) as mask:
            mask = mask.convert("RGB")
            if self.image_size is not None:
                mask = mask.resize(to_pil_size(self.image_size), Image.Resampling.NEAREST)
            mask_array = rgb_to_class(mask).astype(np.int64)

        item = {
            "image": torch.from_numpy(image_array).permute(2, 0, 1).contiguous(),
            "mask": torch.from_numpy(mask_array).long(),
        }

        if self.return_meta:
            item["meta"] = {
                "name": sample.name,
                "image_path": str(sample.image_path),
                "mask_path": str(sample.mask_path),
                "original_size": original_size,
                "image_size": self.image_size,
                "split": self.split,
            }

        if self.transform is not None:
            item = self.transform(item)
        return item


def build_samples(
    root: str | Path,
    split: SplitName,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> list[FundusAVSegSample]:
    """Build paired samples for a requested split."""
    root = Path(root).expanduser().resolve()
    image_dir = root / "images"
    mask_dir = root / "annotation"
    ensure_dataset_layout(root, image_dir, mask_dir)

    train_names = read_split(root / "training.txt")
    test_names = read_split(root / "testing.txt")

    if split in ("train", "val"):
        train_names, val_names = split_train_val(train_names, val_fraction, seed)
        names = train_names if split == "train" else val_names
    elif split == "trainval":
        names = train_names
    elif split == "test":
        names = test_names
    elif split == "all":
        names = sorted(path.name for path in image_dir.glob("*.png"))
    else:
        raise ValueError(f"Unsupported split: {split}")

    samples = [
        FundusAVSegSample(name=name, image_path=image_dir / name, mask_path=mask_dir / name)
        for name in names
    ]
    validate_samples(samples)
    return samples


def read_split(path: Path) -> list[str]:
    """Read a split file containing one image name per line."""
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_train_val(
    names: list[str],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Deterministically split official training names into train/val."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    names = list(names)
    rng = random.Random(seed)
    rng.shuffle(names)

    val_count = int(round(len(names) * val_fraction))
    if val_fraction > 0 and val_count == 0 and len(names) > 1:
        val_count = 1

    val_names = sorted(names[:val_count])
    train_names = sorted(names[val_count:])
    return train_names, val_names


def normalize_image_size(
    image_size: tuple[int, int] | list[int] | None,
) -> tuple[int, int] | None:
    """Normalize image size as (height, width)."""
    if image_size is None:
        return None
    if len(image_size) != 2:
        raise ValueError(f"image_size must have two values, got {image_size}")
    height, width = int(image_size[0]), int(image_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    return height, width


def to_pil_size(image_size: tuple[int, int]) -> tuple[int, int]:
    """Convert (height, width) to PIL's (width, height)."""
    height, width = image_size
    return width, height


def ensure_dataset_layout(root: Path, image_dir: Path, mask_dir: Path) -> None:
    """Validate the expected Fundus-AVSeg directory layout."""
    missing = [path for path in (root, image_dir, mask_dir) if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing Fundus-AVSeg paths: {joined}")


def validate_samples(samples: list[FundusAVSegSample]) -> None:
    """Ensure every sample has both image and mask files."""
    missing = [
        sample.name
        for sample in samples
        if not sample.image_path.exists() or not sample.mask_path.exists()
    ]
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"Missing image/mask files for: {preview}")
