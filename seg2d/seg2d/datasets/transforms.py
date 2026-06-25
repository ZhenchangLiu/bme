"""Tensor transforms for paired 2D segmentation samples."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - exercised when torch is absent.
    torch = None


SampleTransform = Callable[[dict[str, Any]], dict[str, Any]]


class Compose:
    """Compose sample transforms."""

    def __init__(self, transforms: list[SampleTransform]) -> None:
        self.transforms = transforms

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class RandomHorizontalFlip:
    """Randomly flip image and mask along width."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        require_torch()
        if random.random() < self.p:
            sample["image"] = torch.flip(sample["image"], dims=[2])
            sample["mask"] = torch.flip(sample["mask"], dims=[1])
        return sample


class RandomVerticalFlip:
    """Randomly flip image and mask along height."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        require_torch()
        if random.random() < self.p:
            sample["image"] = torch.flip(sample["image"], dims=[1])
            sample["mask"] = torch.flip(sample["mask"], dims=[0])
        return sample


class RandomRotate90:
    """Randomly rotate image and mask by 90, 180, or 270 degrees."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        require_torch()
        if random.random() < self.p:
            k = random.randint(1, 3)
            sample["image"] = torch.rot90(sample["image"], k=k, dims=[1, 2])
            sample["mask"] = torch.rot90(sample["mask"], k=k, dims=[0, 1])
        return sample


class ColorJitter:
    """Random brightness and contrast jitter for image tensors in [0, 1]."""

    def __init__(self, brightness: float = 0.1, contrast: float = 0.1, p: float = 0.8) -> None:
        self.brightness = max(0.0, brightness)
        self.contrast = max(0.0, contrast)
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        require_torch()
        if random.random() >= self.p:
            return sample

        image = sample["image"]
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = image * factor
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = image.mean(dim=(1, 2), keepdim=True)
            image = (image - mean) * factor + mean
        sample["image"] = image.clamp(0.0, 1.0)
        return sample


def build_train_transform(config: dict[str, Any]) -> SampleTransform | None:
    """Build the configured train-time transform pipeline."""
    augmentation = config.get("data", {}).get("augmentation", {})
    if not augmentation or not augmentation.get("enabled", False):
        return None

    transforms: list[SampleTransform] = []
    if augmentation.get("horizontal_flip_p", 0.0) > 0:
        transforms.append(RandomHorizontalFlip(float(augmentation["horizontal_flip_p"])))
    if augmentation.get("vertical_flip_p", 0.0) > 0:
        transforms.append(RandomVerticalFlip(float(augmentation["vertical_flip_p"])))
    if augmentation.get("rotate90_p", 0.0) > 0:
        transforms.append(RandomRotate90(float(augmentation["rotate90_p"])))

    color_jitter = augmentation.get("color_jitter", {})
    if color_jitter and color_jitter.get("enabled", False):
        transforms.append(
            ColorJitter(
                brightness=float(color_jitter.get("brightness", 0.1)),
                contrast=float(color_jitter.get("contrast", 0.1)),
                p=float(color_jitter.get("p", 0.8)),
            )
        )

    if not transforms:
        return None
    return Compose(transforms)


def require_torch() -> None:
    if torch is None:
        raise ImportError("seg2d dataset transforms require PyTorch.")
