"""Track-level event logic for final object passing decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import PipelineTrack


@dataclass(frozen=True)
class ObjectPassingConfig:
    axis: str = "y"
    line_ratio: float = 0.88
    direction: str = "positive"
    trigger_position: str = "leading_edge"
    min_track_age: int = 1
    unknown_label: str = "unknown"


@dataclass(frozen=True)
class ObjectPassedEvent:
    event: str
    track_id: int
    frame_index: int
    label: str
    score: float
    box: tuple[int, int, int, int]
    centroid: tuple[int, int]
    line_position: int
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "label": self.label,
            "score": self.score,
            "box": self.box,
            "centroid": self.centroid,
            "line_position": self.line_position,
            "direction": self.direction,
        }


class ObjectPassingDetector:
    """Emit one event when a tracked centroid crosses a virtual line."""

    def __init__(self, config: ObjectPassingConfig | None = None) -> None:
        self.config = config or ObjectPassingConfig()
        self._last_positions: dict[int, int] = {}
        self._emitted_track_ids: set[int] = set()
        self._validate()

    def update(
        self,
        *,
        frame_index: int,
        frame_shape: tuple[int, int, int],
        tracks: list[PipelineTrack],
        track_labels: dict[int, dict[str, Any]] | None = None,
    ) -> list[ObjectPassedEvent]:
        line_position = self.line_position(frame_shape)
        labels = track_labels or {}
        events: list[ObjectPassedEvent] = []

        for track in tracks:
            position = self._track_position(track)
            previous = self._last_positions.get(track.track_id)
            self._last_positions[track.track_id] = position

            if track.track_id in self._emitted_track_ids:
                continue
            if track.age < self.config.min_track_age:
                continue
            if not self._has_crossed(previous, position, line_position):
                continue

            prediction = labels.get(track.track_id, {})
            event = ObjectPassedEvent(
                event="object_passed",
                track_id=track.track_id,
                frame_index=frame_index,
                label=str(prediction.get("label", self.config.unknown_label)),
                score=float(prediction.get("score", 0.0)),
                box=track.box,
                centroid=track.centroid,
                line_position=line_position,
                direction=self.config.direction,
            )
            self._emitted_track_ids.add(track.track_id)
            events.append(event)

        return events

    def line_position(self, frame_shape: tuple[int, int, int]) -> int:
        height, width = frame_shape[:2]
        size = height if self.config.axis == "y" else width
        return int(size * self.config.line_ratio)

    def reset(self) -> None:
        self._last_positions.clear()
        self._emitted_track_ids.clear()

    def _has_crossed(self, previous: int | None, current: int, line_position: int) -> bool:
        if previous is None:
            return current >= line_position if self.config.direction == "positive" else current <= line_position
        if self.config.direction == "positive":
            return previous < line_position <= current
        return previous > line_position >= current

    def _validate(self) -> None:
        if self.config.axis not in {"x", "y"}:
            raise ValueError("event axis must be 'x' or 'y'")
        if self.config.direction not in {"positive", "negative"}:
            raise ValueError("event direction must be 'positive' or 'negative'")
        if self.config.trigger_position not in {"centroid", "leading_edge", "trailing_edge"}:
            raise ValueError("event trigger_position must be 'centroid', 'leading_edge', or 'trailing_edge'")
        if not 0.0 <= self.config.line_ratio <= 1.0:
            raise ValueError("event line_ratio must be between 0.0 and 1.0")

    def _track_position(self, track: PipelineTrack) -> int:
        if self.config.trigger_position == "centroid":
            return track.centroid[1] if self.config.axis == "y" else track.centroid[0]

        x, y, width, height = track.box
        if self.config.axis == "y":
            leading_edge = y + height if self.config.direction == "positive" else y
            trailing_edge = y if self.config.direction == "positive" else y + height
        else:
            leading_edge = x + width if self.config.direction == "positive" else x
            trailing_edge = x if self.config.direction == "positive" else x + width

        if self.config.trigger_position == "leading_edge":
            return int(leading_edge)
        return int(trailing_edge)
