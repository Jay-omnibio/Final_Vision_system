"""Detector + tracker pipeline runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import VisionPipelineConfig
from .project_paths import ensure_model_import_paths
from .types import FrameResult, PipelineDetection, PipelineTrack

ensure_model_import_paths()

from object_tracker import CentroidTracker  # noqa: E402


class VisionPipeline:
    """Load detector and tracker once, then process frames repeatedly."""

    def __init__(self, config: VisionPipelineConfig | None = None) -> None:
        self.config = config or VisionPipelineConfig()
        self.detector = self._create_detector()
        self.tracker = CentroidTracker(
            max_distance=self.config.max_track_distance,
            max_missing_frames=self.config.max_missing_frames,
            prefer_downward_motion=self.config.prefer_downward_motion,
            start_track_id=self.config.start_track_id,
        )
        self.frame_index = 0

    @classmethod
    def from_yaml(cls, path: str) -> "VisionPipeline":
        """Create a pipeline from a YAML config file."""
        return cls(VisionPipelineConfig.from_yaml(path))

    def process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        """Detect and track one BGR frame."""
        self.frame_index += 1
        raw_boxes = self.detector.detect_frame(frame_bgr)
        detections = [self._to_detection(box) for box in raw_boxes]
        tracks = self.tracker.update([detection.box for detection in detections])
        return FrameResult(
            frame_index=self.frame_index,
            detections=detections,
            tracks=[self._to_track(track) for track in tracks],
            detector_type="yolo",
        )

    def reset_tracker(self, *, reset_ids: bool = False) -> None:
        """Clear tracker state and optionally restart IDs from 1."""
        self.tracker.clear(reset_ids=reset_ids)
        if reset_ids:
            self.frame_index = 0

    def _create_detector(self):
        from yolo_detector import YoloDetector

        return YoloDetector(
            weights=self.config.yolo_weights,
            conf=self.config.yolo_conf,
            iou=self.config.yolo_iou,
            device=self.config.yolo_device,
        )

    @staticmethod
    def _to_detection(box: Any) -> PipelineDetection:
        return PipelineDetection(
            x=int(box.x),
            y=int(box.y),
            width=int(box.width),
            height=int(box.height),
            score=float(getattr(box, "score", 1.0)),
            class_id=getattr(box, "class_id", None),
            class_name=getattr(box, "class_name", None),
        )

    @staticmethod
    def _to_track(track: Any) -> PipelineTrack:
        x, y, width, height = track.box
        return PipelineTrack(
            track_id=int(track.track_id),
            x=int(x),
            y=int(y),
            width=int(width),
            height=int(height),
            centroid=tuple(track.centroid),
            age=int(track.age),
            missing_frames=int(track.missing_frames),
            history=[tuple(point) for point in track.history],
        )
