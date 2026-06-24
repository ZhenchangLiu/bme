"""Mask color mapping utilities for Fundus-AVSeg."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
from PIL import Image


CLASS_NAMES = {
    0: "background",
    1: "artery",
    2: "vein",
    3: "crossing",
    4: "uncertain",
}

CLASS_TO_COLOR = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (0, 255, 0),
    4: (255, 255, 255),
}

COLOR_TO_CLASS = {color: class_id for class_id, color in CLASS_TO_COLOR.items()}
VESSEL_CLASS_IDS = (1, 2, 3, 4)


def as_rgb_array(mask: Image.Image | np.ndarray) -> np.ndarray:
    """Return an RGB uint8 array from a PIL image or numpy mask."""
    if isinstance(mask, Image.Image):
        array = np.asarray(mask.convert("RGB"))
    else:
        array = np.asarray(mask)
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError(f"Expected HxW, HxWx3, or HxWx4 mask, got {array.shape}")
        array = array[..., :3]

    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return array


def rgb_to_class(mask: Image.Image | np.ndarray, unknown_value: int | None = None) -> np.ndarray:
    """Convert a Fundus-AVSeg RGB annotation to a class-id mask.

    Args:
        mask: RGB annotation image.
        unknown_value: Value assigned to unknown colors. If omitted, unknown
            colors raise a ValueError.

    Returns:
        A uint8 array with shape HxW.
    """
    rgb = as_rgb_array(mask)
    class_mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)

    for color, class_id in COLOR_TO_CLASS.items():
        matches = np.all(rgb == color, axis=-1)
        class_mask[matches] = class_id

    unknown = class_mask == 255
    if np.any(unknown):
        if unknown_value is None:
            unknown_colors = count_colors(rgb[unknown])
            preview = ", ".join(f"{color}: {count}" for color, count in unknown_colors[:10])
            raise ValueError(f"Unknown mask colors found: {preview}")
        class_mask[unknown] = unknown_value

    return class_mask


def class_to_rgb(mask: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a class-id mask to an RGB visualization mask."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Expected HxW class mask, got {array.shape}")

    rgb = np.zeros((*array.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_TO_COLOR.items():
        rgb[array == class_id] = color
    return rgb


def vessel_mask(mask: Image.Image | np.ndarray, class_ids: Iterable[int] = VESSEL_CLASS_IDS) -> np.ndarray:
    """Return a binary vessel mask from RGB or class-id input."""
    array = np.asarray(mask)
    if array.ndim == 3:
        array = rgb_to_class(array)
    return np.isin(array, tuple(class_ids)).astype(np.uint8)


def count_colors(rgb_pixels: np.ndarray) -> list[tuple[tuple[int, int, int], int]]:
    """Count RGB colors and return them sorted by frequency."""
    pixels = np.asarray(rgb_pixels, dtype=np.uint8).reshape(-1, 3)
    counter: Counter[tuple[int, int, int]] = Counter(map(tuple, pixels.tolist()))
    return counter.most_common()
