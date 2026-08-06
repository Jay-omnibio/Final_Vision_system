"""Shared YOLO detector result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    width: int
    height: int
    score: float = 1.0
    class_id: int | None = None
    class_name: str | None = None

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
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass(frozen=True)
class DetectionResult:
    boxes: list[BBox]
    image_width: int
    image_height: int
    model_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "boxes": [box.to_dict() for box in self.boxes],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "model_path": self.model_path,
        }

