"""Image helpers for subtract detector."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .types import BBox


def load_image_bgr(image: str | Path | Image.Image | np.ndarray, *, array_format: str = "bgr") -> np.ndarray:
    """Load path/PIL/NumPy input and return a BGR NumPy image."""
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image array, got shape {image.shape}")
        if array_format == "rgb":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if array_format == "bgr":
            return image.copy()
        raise ValueError("array_format must be 'rgb' or 'bgr'")

    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    frame = cv2.imread(str(Path(image)))
    if frame is None:
        raise FileNotFoundError(f"Could not load image: {image}")
    return frame


def save_image(path: str | Path, frame_bgr: np.ndarray) -> Path:
    """Save a BGR image and return the output path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame_bgr):
        raise OSError(f"Could not write image: {output}")
    return output


def draw_boxes(image_bgr: np.ndarray, boxes: list[BBox]) -> np.ndarray:
    """Draw boxes on a BGR image."""
    output = image_bgr.copy()
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 80), 2)
        cv2.putText(
            output,
            f"box {index}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 80),
            2,
            cv2.LINE_AA,
        )
    return output

