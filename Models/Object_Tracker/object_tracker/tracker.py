"""Centroid tracker for frame-to-frame bounding boxes."""

from __future__ import annotations

from .types import Box, Point, TrackedObject


def box_centroid(box: Box) -> Point:
    x, y, width, height = box
    return x + width // 2, y + height // 2


def centroid_distance(first: Point, second: Point) -> float:
    dx = first[0] - second[0]
    dy = first[1] - second[1]
    return (dx * dx + dy * dy) ** 0.5


class CentroidTracker:
    """Assign stable track IDs to bounding boxes across frames."""

    def __init__(
        self,
        *,
        max_distance: float = 80.0,
        max_missing_frames: int = 8,
        prefer_downward_motion: bool = True,
        start_track_id: int = 1,
        max_history: int = 12,
        upward_tolerance: int = 15,
    ) -> None:
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.prefer_downward_motion = prefer_downward_motion
        self.max_history = max_history
        self.upward_tolerance = upward_tolerance
        self._next_track_id = max(1, int(start_track_id))
        self._tracks: dict[int, TrackedObject] = {}

    @property
    def tracks(self) -> list[TrackedObject]:
        return sorted(self._tracks.values(), key=lambda track: track.track_id)

    @property
    def next_track_id(self) -> int:
        return self._next_track_id

    def clear(self, *, reset_ids: bool = False) -> None:
        """Drop active tracks and optionally reset the ID counter."""
        self._tracks.clear()
        if reset_ids:
            self._next_track_id = 1

    def update(self, boxes: list[Box]) -> list[TrackedObject]:
        """Update active tracks from the current frame boxes."""
        detections = [(box, box_centroid(box)) for box in boxes]

        if not self._tracks:
            for box, centroid in detections:
                self._register(box, centroid)
            return self.tracks

        unmatched_track_ids = set(self._tracks)
        unmatched_detection_indexes = set(range(len(detections)))
        matches = self._match_tracks(detections)

        for track_id, detection_index in matches:
            box, centroid = detections[detection_index]
            track = self._tracks[track_id]
            track.box = box
            track.centroid = centroid
            track.age += 1
            track.missing_frames = 0
            track.history.append(centroid)
            track.history = track.history[-self.max_history :]
            unmatched_track_ids.discard(track_id)
            unmatched_detection_indexes.discard(detection_index)

        for track_id in list(unmatched_track_ids):
            track = self._tracks[track_id]
            track.missing_frames += 1
            if track.missing_frames > self.max_missing_frames:
                del self._tracks[track_id]

        for detection_index in sorted(unmatched_detection_indexes):
            box, centroid = detections[detection_index]
            self._register(box, centroid)

        return self.tracks

    def _register(self, box: Box, centroid: Point) -> None:
        track = TrackedObject(
            track_id=self._next_track_id,
            box=box,
            centroid=centroid,
            history=[centroid],
        )
        self._tracks[track.track_id] = track
        self._next_track_id += 1

    def _match_tracks(
        self,
        detections: list[tuple[Box, Point]],
    ) -> list[tuple[int, int]]:
        candidates: list[tuple[float, int, int]] = []

        for track in self._tracks.values():
            for detection_index, (_, centroid) in enumerate(detections):
                distance = centroid_distance(track.centroid, centroid)
                if distance > self.max_distance:
                    continue
                if self.prefer_downward_motion and centroid[1] < track.centroid[1] - self.upward_tolerance:
                    continue
                candidates.append((distance, track.track_id, detection_index))

        candidates.sort(key=lambda item: item[0])
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        matches: list[tuple[int, int]] = []

        for _, track_id, detection_index in candidates:
            if track_id in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(detection_index)
            matches.append((track_id, detection_index))

        return matches

