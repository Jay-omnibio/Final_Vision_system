"""Shared types for background subtraction detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    width: int
    height: int
    score: float = 1.0

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "score": self.score,
        }


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    width: int
    height: int
    area: float

    @property
    def box(self) -> Rect:
        return self.x, self.y, self.width, self.height


@dataclass
class BackgroundSubtractionConfig:
    threshold: int = 30
    min_area: int = 500
    kernel_size: int = 25
    roi: Rect | None = None
    resize_frame_to_background: bool = True
    merge_boxes: bool = True
    merge_iou_threshold: float = 0.0
    merge_gap: int = 0
    merge_horizontal_gap: int = 25
    merge_vertical_overlap_ratio: float = 0.35
    merge_vertical_gap: int = 25
    merge_horizontal_overlap_ratio: float = 0.35
    hsv_threshold_boost: int = 8

