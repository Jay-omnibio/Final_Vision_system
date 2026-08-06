"""Configuration for the detector + tracker pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _resolve_optional_path(value: Any, *, base_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return base_dir / path


def _optional_roi(value: Any) -> tuple[int, int, int, int] | None:
    if value in (None, ""):
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("subtract.roi must be null or [x, y, width, height]")
    return tuple(int(part) for part in value)


@dataclass
class VisionPipelineConfig:
    detector_type: str = "yolo"
    yolo_weights: str | Path | None = None
    yolo_conf: float = 0.25
    yolo_iou: float = 0.45
    yolo_device: str | int | None = None

    background_image: str | Path | None = None
    subtract_mode: str = "improved"
    subtract_threshold: int = 30
    subtract_min_area: int = 500
    subtract_kernel_size: int = 25
    subtract_roi: tuple[int, int, int, int] | None = None
    subtract_merge_horizontal_gap: int = 25
    subtract_merge_vertical_overlap_ratio: float = 0.35
    subtract_hsv_threshold_boost: int = 8

    max_track_distance: float = 300.0
    max_missing_frames: int = 4
    prefer_downward_motion: bool = True
    start_track_id: int = 1

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VisionPipelineConfig":
        """Load pipeline config from a YAML file."""
        import yaml

        config_path = Path(path)
        with config_path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError("Config YAML must contain a mapping at the top level")

        base_dir = config_path.resolve().parent
        vision = payload.get("vision", payload)
        if not isinstance(vision, dict):
            raise ValueError("Config 'vision' section must be a mapping")

        yolo = vision.get("yolo", {}) or {}
        subtract = vision.get("subtract", {}) or {}
        tracker = vision.get("tracker", {}) or {}

        return cls(
            detector_type=str(vision.get("detector_type", "yolo")),
            yolo_weights=_resolve_optional_path(yolo.get("weights"), base_dir=base_dir),
            yolo_conf=float(yolo.get("conf", 0.25)),
            yolo_iou=float(yolo.get("iou", 0.45)),
            yolo_device=yolo.get("device"),
            background_image=_resolve_optional_path(subtract.get("background_image"), base_dir=base_dir),
            subtract_mode=str(subtract.get("mode", "improved")),
            subtract_threshold=int(subtract.get("threshold", 30)),
            subtract_min_area=int(subtract.get("min_area", 500)),
            subtract_kernel_size=int(subtract.get("kernel_size", 25)),
            subtract_roi=_optional_roi(subtract.get("roi")),
            subtract_merge_horizontal_gap=int(subtract.get("merge_horizontal_gap", 25)),
            subtract_merge_vertical_overlap_ratio=float(
                subtract.get("merge_vertical_overlap_ratio", 0.35)
            ),
            subtract_hsv_threshold_boost=int(subtract.get("hsv_threshold_boost", 8)),
            max_track_distance=float(tracker.get("max_distance", 300.0)),
            max_missing_frames=int(tracker.get("max_missing_frames", 4)),
            prefer_downward_motion=bool(tracker.get("prefer_downward_motion", True)),
            start_track_id=int(tracker.get("start_track_id", 1)),
        )
