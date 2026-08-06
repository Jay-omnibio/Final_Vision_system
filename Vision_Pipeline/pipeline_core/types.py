"""Pipeline result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PipelineDetection:
    x: int
    y: int
    width: int
    height: int
    score: float = 1.0
    class_id: int | None = None
    class_name: str | None = None

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

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
class PipelineTrack:
    track_id: int
    x: int
    y: int
    width: int
    height: int
    centroid: tuple[int, int]
    age: int
    missing_frames: int
    history: list[tuple[int, int]]

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "box": self.box,
            "centroid": self.centroid,
            "age": self.age,
            "missing_frames": self.missing_frames,
            "history": [tuple(point) for point in self.history],
        }


@dataclass(frozen=True)
class FrameResult:
    frame_index: int
    detections: list[PipelineDetection]
    tracks: list[PipelineTrack]
    detector_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "detector_type": self.detector_type,
            "detections": [detection.to_dict() for detection in self.detections],
            "tracks": [track.to_dict() for track in self.tracks],
        }

