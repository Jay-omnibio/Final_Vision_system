"""Image loading and conversion helpers for DINO embedding."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(image: str | Path | Image.Image | np.ndarray, *, array_format: str = "rgb") -> Image.Image:
    """Load path/PIL/NumPy input and return an RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image array, got shape {image.shape}")
        if array_format == "bgr":
            image = image[:, :, ::-1]
        elif array_format != "rgb":
            raise ValueError("array_format must be 'rgb' or 'bgr'")
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")

    path = Path(image)
    with Image.open(path) as opened:
        return opened.convert("RGB")


def iter_image_paths(folder: str | Path, *, recursive: bool = True) -> Iterable[Path]:
    """Yield supported image files from a folder in stable sorted order."""
    root = Path(folder)
    paths = root.rglob("*") if recursive else root.glob("*")
    for path in sorted(paths):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path

