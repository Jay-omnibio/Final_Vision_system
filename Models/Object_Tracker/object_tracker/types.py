"""Shared object tracking types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Box = tuple[int, int, int, int]
Point = tuple[int, int]


@dataclass
class TrackedObject:
    track_id: int
    box: Box
    centroid: Point
    age: int = 1
    missing_frames: int = 0
    history: list[Point] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "box": tuple(self.box),
            "centroid": tuple(self.centroid),
            "age": self.age,
            "missing_frames": self.missing_frames,
            "history": [tuple(point) for point in self.history],
        }
