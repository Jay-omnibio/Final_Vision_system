"""Background-subtraction detector wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .core import create_improved_mask, create_standard_mask, find_detections, merge_detections, resize_like
from .image_utils import load_image_bgr
from .types import BBox, BackgroundSubtractionConfig


class SubtractDetector:
    """Detect foreground boxes from an empty-background image."""

    def __init__(
        self,
        background: np.ndarray,
        *,
        mode: str = "improved",
        config: BackgroundSubtractionConfig | None = None,
    ) -> None:
        if mode not in {"standard", "improved"}:
            raise ValueError("mode must be 'standard' or 'improved'")
        self.background = background
        self.mode = mode
        self.config = config or BackgroundSubtractionConfig()

    @classmethod
    def from_background_image(
        cls,
        background: str | Path | Image.Image | np.ndarray,
        *,
        mode: str = "improved",
        config: BackgroundSubtractionConfig | None = None,
        array_format: str = "bgr",
    ) -> "SubtractDetector":
        return cls(
            load_image_bgr(background, array_format=array_format),
            mode=mode,
            config=config,
        )

    def detect_image(
        self,
        image: str | Path | Image.Image | np.ndarray,
        *,
        array_format: str = "bgr",
    ) -> list[BBox]:
        frame = load_image_bgr(image, array_format=array_format)
        return self.detect_frame(frame)

    def detect_frame(self, frame_bgr: np.ndarray) -> list[BBox]:
        working = frame_bgr
        if frame_bgr.shape != self.background.shape and self.config.resize_frame_to_background:
            working = resize_like(frame_bgr, self.background)

        if self.mode == "standard":
            mask = create_standard_mask(working, self.background, self.config)
        else:
            mask = create_improved_mask(working, self.background, self.config)

        detections = find_detections(mask, self.config.min_area)
        if self.config.merge_boxes:
            detections = merge_detections(detections, self.config)

        offset_x = self.config.roi[0] if self.config.roi else 0
        offset_y = self.config.roi[1] if self.config.roi else 0
        return [
            BBox(
                x=detection.x + offset_x,
                y=detection.y + offset_y,
                width=detection.width,
                height=detection.height,
            )
            for detection in detections
        ]

    def foreground_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return the binary foreground mask for debugging."""
        working = frame_bgr
        if frame_bgr.shape != self.background.shape and self.config.resize_frame_to_background:
            working = resize_like(frame_bgr, self.background)
        if self.mode == "standard":
            return create_standard_mask(working, self.background, self.config)
        return create_improved_mask(working, self.background, self.config)

